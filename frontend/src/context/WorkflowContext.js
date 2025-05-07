import React, { createContext, useContext, useReducer, useEffect } from 'react';
import WebSocketService from '../services/websocketService';

// État initial
const initialState = {
  workflows: [],
  activeWorkflow: null,
  tasks: [],
  results: null,
  dashboardData: null,
  loading: false,
  error: null,
  notifications: [],
};

// Actions
const ACTIONS = {
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  SET_WORKFLOWS: 'SET_WORKFLOWS',
  SET_ACTIVE_WORKFLOW: 'SET_ACTIVE_WORKFLOW',
  SET_TASKS: 'SET_TASKS',
  SET_RESULTS: 'SET_RESULTS',
  SET_DASHBOARD_DATA: 'SET_DASHBOARD_DATA',
  ADD_NOTIFICATION: 'ADD_NOTIFICATION',
  REMOVE_NOTIFICATION: 'REMOVE_NOTIFICATION',
  UPDATE_WORKFLOW_STATUS: 'UPDATE_WORKFLOW_STATUS',
  UPDATE_TASK_STATUS: 'UPDATE_TASK_STATUS',
};

// Réducteur
const workflowReducer = (state, action) => {
  switch (action.type) {
    case ACTIONS.SET_LOADING:
      return { ...state, loading: action.payload };
    case ACTIONS.SET_ERROR:
      return { ...state, error: action.payload, loading: false };
    case ACTIONS.SET_WORKFLOWS:
      return { ...state, workflows: action.payload, loading: false };
    case ACTIONS.SET_ACTIVE_WORKFLOW:
      return { ...state, activeWorkflow: action.payload, loading: false };
    case ACTIONS.SET_TASKS:
      return { ...state, tasks: action.payload, loading: false };
    case ACTIONS.SET_RESULTS:
      return { ...state, results: action.payload, loading: false };
    case ACTIONS.SET_DASHBOARD_DATA:
      return { ...state, dashboardData: action.payload, loading: false };
    case ACTIONS.ADD_NOTIFICATION:
      return { 
        ...state, 
        notifications: [...state.notifications, action.payload] 
      };
    case ACTIONS.REMOVE_NOTIFICATION:
      return { 
        ...state, 
        notifications: state.notifications.filter(n => n.id !== action.payload) 
      };
    case ACTIONS.UPDATE_WORKFLOW_STATUS:
      return {
        ...state,
        workflows: state.workflows.map(wf => 
          wf.id === action.payload.id 
            ? { ...wf, status: action.payload.status } 
            : wf
        ),
        activeWorkflow: state.activeWorkflow && state.activeWorkflow.id === action.payload.id
          ? { ...state.activeWorkflow, status: action.payload.status }
          : state.activeWorkflow
      };
    case ACTIONS.UPDATE_TASK_STATUS:
      return {
        ...state,
        tasks: state.tasks.map(task => 
          task.id === action.payload.id 
            ? { ...task, status: action.payload.status, progress: action.payload.progress } 
            : task
        )
      };
    default:
      return state;
  }
};

// Contexte
const WorkflowContext = createContext();

// Provider
export const WorkflowProvider = ({ children }) => {
  const [state, dispatch] = useReducer(workflowReducer, initialState);
  const webSocketService = new WebSocketService();
  
  useEffect(() => {
    // Connexion WebSocket
    webSocketService.connect();
    
    // Écouteurs WebSocket
    const unsubscribeWorkflowUpdate = webSocketService.subscribe('workflow_update', (data) => {
      dispatch({
        type: ACTIONS.UPDATE_WORKFLOW_STATUS,
        payload: { id: data.workflow_id, status: data.status }
      });
      
      dispatch({
        type: ACTIONS.ADD_NOTIFICATION,
        payload: {
          id: Date.now(),
          message: `Workflow ${data.workflow_name} mis à jour: ${data.status}`,
          type: 'info',
          timestamp: new Date()
        }
      });
    });
    
    const unsubscribeTaskUpdate = webSocketService.subscribe('task_update', (data) => {
      dispatch({
        type: ACTIONS.UPDATE_TASK_STATUS,
        payload: { 
          id: data.task_id, 
          status: data.status, 
          progress: data.progress 
        }
      });
    });
    
    const unsubscribeError = webSocketService.subscribe('error', (data) => {
      dispatch({
        type: ACTIONS.ADD_NOTIFICATION,
        payload: {
          id: Date.now(),
          message: data.message,
          type: 'error',
          timestamp: new Date()
        }
      });
    });
    
    return () => {
      unsubscribeWorkflowUpdate();
      unsubscribeTaskUpdate();
      unsubscribeError();
      webSocketService.disconnect();
    };
  }, []);
  
  return (
    <WorkflowContext.Provider value={{ state, dispatch, actions: ACTIONS, webSocketService }}>
      {children}
    </WorkflowContext.Provider>
  );
};

// Hook personnalisé
export const useWorkflow = () => {
  const context = useContext(WorkflowContext);
  if (!context) {
    throw new Error('useWorkflow must be used within a WorkflowProvider');
  }
  return context;
};