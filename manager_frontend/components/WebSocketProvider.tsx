

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

interface WebSocketContextType {
  isConnected: boolean;
  connectionError: string | null;
  lastMessage: any;
  subscribeToWorkflow: (workflowId: string) => void;
  subscribeToTask: (taskId: string) => void;
  subscribeToVolunteer: (volunteerId: string) => void;
  forceReconnect: () => void;
  getConnectionState: () => string;
  workflows: any[];
  tasks: any[];
  volunteers: any[];
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
  onWorkflowUpdate?: (data: any) => void;
  onWorkflowStatusChange?: (data: any) => void;
  onTaskUpdate?: (data: any) => void;
  onTaskProgress?: (data: any) => void;
  onVolunteerUpdate?: (data: any) => void;
  onVolunteerStatus?: (data: any) => void;
  onCustomEvent?: (type: string, data: any) => void;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  children,
  onWorkflowUpdate,
  onWorkflowStatusChange,
  onTaskUpdate,
  onTaskProgress,
  onVolunteerUpdate,
  onVolunteerStatus,
  onCustomEvent,
}) => {
  const [workflows, setWorkflows] = useState<any[]>([]);
  const [tasks, setTasks] = useState<any[]>([]);
  const [volunteers, setVolunteers] = useState<any[]>([]);

  // Gestionnaires globaux pour mettre à jour l'état
  const handleWorkflowUpdate = (data: any) => {
    console.log('Mise à jour workflow reçue:', data);
    
    // Mettre à jour l'état global
    if (data.workflow) {
      setWorkflows(prev => {
        const workflowIndex = prev.findIndex(w => w.id === data.workflow.id);
        if (workflowIndex >= 0) {
          const updated = [...prev];
          updated[workflowIndex] = { ...updated[workflowIndex], ...data.workflow };
          return updated;
        } else if (data.action === 'created') {
          return [...prev, data.workflow];
        }
        return prev;
      });
    }

    // Appeler le gestionnaire personnalisé si fourni
    onWorkflowUpdate?.(data);
  };

  const handleWorkflowStatusChange = (data: any) => {
    console.log('Changement statut workflow reçu:', data);
    
    // Mettre à jour l'état global
    if (data.workflow_id) {
      setWorkflows(prev => {
        const workflowIndex = prev.findIndex(w => w.id === data.workflow_id);
        if (workflowIndex >= 0) {
          const updated = [...prev];
          updated[workflowIndex] = { 
            ...updated[workflowIndex], 
            status: data.status,
            statusMessage: data.message
          };
          return updated;
        }
        return prev;
      });
    }

    // Appeler le gestionnaire personnalisé si fourni
    onWorkflowStatusChange?.(data);
  };

  const handleTaskUpdate = (data: any) => {
    console.log('Mise à jour tâche reçue:', data);
    
    // Mettre à jour l'état global
    if (data.task) {
      setTasks(prev => {
        const taskIndex = prev.findIndex(t => t.id === data.task.id);
        if (taskIndex >= 0) {
          const updated = [...prev];
          updated[taskIndex] = { ...updated[taskIndex], ...data.task };
          return updated;
        } else if (data.action === 'created') {
          return [...prev, data.task];
        }
        return prev;
      });
    }

    // Appeler le gestionnaire personnalisé si fourni
    onTaskUpdate?.(data);
  };

  const handleTaskProgress = (data: any) => {
    console.log('Progression tâche reçue:', data);
    
    // Mettre à jour l'état global des tâches
    if (data.task_id) {
      setTasks(prev => {
        const taskIndex = prev.findIndex(t => t.id === data.task_id);
        if (taskIndex >= 0) {
          const updated = [...prev];
          updated[taskIndex] = { 
            ...updated[taskIndex], 
            progress: data.progress,
            lastProgressUpdate: new Date().toISOString()
          };
          return updated;
        }
        return prev;
      });
    }

    // Appeler le gestionnaire personnalisé si fourni
    onTaskProgress?.(data);
  };

  const handleVolunteerUpdate = (data: any) => {
    console.log('Mise à jour volontaire reçue:', data);
    
    // Mettre à jour l'état global
    if (data.volunteer) {
      setVolunteers(prev => {
        const volunteerIndex = prev.findIndex(v => v.id === data.volunteer.id);
        if (volunteerIndex >= 0) {
          const updated = [...prev];
          updated[volunteerIndex] = { ...updated[volunteerIndex], ...data.volunteer };
          return updated;
        } else if (data.action === 'registered') {
          return [...prev, data.volunteer];
        }
        return prev;
      });
    }

    // Appeler le gestionnaire personnalisé si fourni
    onVolunteerUpdate?.(data);
  };

  const handleVolunteerStatus = (data: any) => {
    console.log('Statut volontaire reçu:', data);
    
    // Mettre à jour l'état global
    if (data.volunteer_id) {
      setVolunteers(prev => {
        const volunteerIndex = prev.findIndex(v => v.id === data.volunteer_id);
        if (volunteerIndex >= 0) {
          const updated = [...prev];
          updated[volunteerIndex] = { 
            ...updated[volunteerIndex], 
            status: data.status !== undefined ? data.status : updated[volunteerIndex].status,
            available: data.available !== undefined ? data.available : updated[volunteerIndex].available,
            lastStatusUpdate: new Date().toISOString()
          };
          return updated;
        }
        return prev;
      });
    }

    // Appeler le gestionnaire personnalisé si fourni
    onVolunteerStatus?.(data);
  };

  const handleCustomEvent = (type: string, data: any) => {
    console.log('Événement personnalisé reçu:', type, data);
    
    // Traiter les événements personnalisés spéciaux
    switch (type) {
      case 'workflow_status_change':
        handleWorkflowStatusChange(data);
        break;
      case 'multiple_updates':
        // Traiter plusieurs mises à jour en une fois
        if (data.workflows) {
          data.workflows.forEach((workflow: any) => {
            handleWorkflowUpdate({ workflow, action: 'updated' });
          });
        }
        if (data.tasks) {
          data.tasks.forEach((task: any) => {
            handleTaskUpdate({ task, action: 'updated' });
          });
        }
        if (data.volunteers) {
          data.volunteers.forEach((volunteer: any) => {
            handleVolunteerUpdate({ volunteer, action: 'updated' });
          });
        }
        break;
      default:
        // Événement personnalisé générique
        break;
    }

    // Appeler le gestionnaire personnalisé si fourni
    onCustomEvent?.(type, data);
  };

  const {
    isConnected,
    connectionError,
    lastMessage,
    subscribeToWorkflow,
    subscribeToTask,
    subscribeToVolunteer,
    forceReconnect,
    getConnectionState,
  } = useWebSocket({
    onWorkflowUpdate: handleWorkflowUpdate,
    onWorkflowStatusChange: handleWorkflowStatusChange,
    onTaskUpdate: handleTaskUpdate,
    onTaskProgress: handleTaskProgress,
    onVolunteerUpdate: handleVolunteerUpdate,
    onVolunteerStatus: handleVolunteerStatus,
    onCustomEvent: handleCustomEvent,
    autoConnect: true,
  });

  // Afficher les informations de connexion dans la console
  useEffect(() => {
    if (isConnected) {
      console.log('WebSocketProvider: Connecté avec succès');
    } else if (connectionError) {
      console.error('WebSocketProvider: Erreur de connexion:', connectionError);
    }
  }, [isConnected, connectionError]);

  // Log des changements d'état
  useEffect(() => {
    console.log('WebSocketProvider: État des workflows:', workflows.length);
  }, [workflows]);

  useEffect(() => {
    console.log('WebSocketProvider: État des tâches:', tasks.length);
  }, [tasks]);

  useEffect(() => {
    console.log('WebSocketProvider: État des volontaires:', volunteers.length);
  }, [volunteers]);

  const contextValue: WebSocketContextType = React.useMemo(() => ({
    isConnected,
    connectionError,
    lastMessage,
    subscribeToWorkflow,
    subscribeToTask,
    subscribeToVolunteer,
    forceReconnect,
    getConnectionState,
    workflows,
    tasks,
    volunteers,
  }), [isConnected, connectionError, lastMessage, subscribeToWorkflow, subscribeToTask, subscribeToVolunteer, forceReconnect, getConnectionState, workflows, tasks, volunteers]);

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocketContext = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider');
  }
  return context;
};

// Hook spécialisé pour les workflows
export const useWorkflowWebSocket = (workflowId?: string) => {
  const context = useWebSocketContext();
  
  useEffect(() => {
    if (workflowId && context.isConnected) {
      context.subscribeToWorkflow(workflowId);
    }
  }, [workflowId, context.isConnected, context.subscribeToWorkflow]);
  
  return {
    ...context,
    currentWorkflow: workflowId ? context.workflows.find(w => w.id === workflowId) : null,
    workflowTasks: workflowId ? context.tasks.filter(t => t.workflow_id === workflowId || t.workflow === workflowId) : [],
  };
};

// Hook spécialisé pour les tâches
export const useTaskWebSocket = (taskId?: string) => {
  const context = useWebSocketContext();
  
  useEffect(() => {
    if (taskId && context.isConnected) {
      context.subscribeToTask(taskId);
    }
  }, [taskId, context.isConnected, context.subscribeToTask]);
  
  return {
    ...context,
    currentTask: taskId ? context.tasks.find(t => t.id === taskId) : null,
  };
};

// Hook spécialisé pour les volontaires
export const useVolunteerWebSocket = (volunteerId?: string) => {
  const context = useWebSocketContext();
  
  useEffect(() => {
    if (volunteerId && context.isConnected) {
      context.subscribeToVolunteer(volunteerId);
    }
  }, [volunteerId, context.isConnected, context.subscribeToVolunteer]);
  
  return {
    ...context,
    currentVolunteer: volunteerId ? context.volunteers.find(v => v.id === volunteerId) : null,
  };
};