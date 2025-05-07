// backend/src/services/WebSocketService.js

import ReconnectingWebSocket from 'reconnecting-websocket';

export default class WebSocketService {
  constructor() {
    this.callbacks = {};
    this.socket = null;
    this.connected = false;
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = localStorage.getItem('auth_token');
    const wsUrl = `${protocol}//${window.location.host}/ws/matrix-workflows/?token=${token}`;
    
    this.socket = new ReconnectingWebSocket(wsUrl);
    
    this.socket.onopen = () => {
      console.log('WebSocket connection established');
      this.connected = true;
      
      if (this.callbacks['connect']) {
        this.callbacks['connect'].forEach(callback => callback());
      }
    };
    
    this.socket.onclose = (e) => {
      console.log('WebSocket connection closed:', e);
      this.connected = false;
      
      if (this.callbacks['disconnect']) {
        this.callbacks['disconnect'].forEach(callback => callback(e));
      }
    };
    
    this.socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const messageType = data.type;
      
      if (this.callbacks[messageType]) {
        this.callbacks[messageType].forEach(callback => callback(data));
      }
      
      if (this.callbacks['message']) {
        this.callbacks['message'].forEach(callback => callback(data));
      }
      
      // Traitement spécifique pour les mises à jour de tâches matricielles
      if (messageType === 'task_update') {
        if (this.callbacks['task_status_' + data.task_id]) {
          this.callbacks['task_status_' + data.task_id].forEach(callback => callback(data));
        }
      }
      
      // Traitement spécifique pour les mises à jour de workflow matriciel
      if (messageType === 'workflow_update') {
        if (this.callbacks['workflow_status_' + data.workflow_id]) {
          this.callbacks['workflow_status_' + data.workflow_id].forEach(callback => callback(data));
        }
      }
      
      // Traitement spécifique pour les mises à jour du Coordinateur
      if (messageType === 'coordinator_update') {
        if (this.callbacks['coordinator_status']) {
          this.callbacks['coordinator_status'].forEach(callback => callback(data));
        }
      }
      
      // Traitement spécifique pour les mises à jour des volontaires
      if (messageType === 'volunteers_update') {
        if (this.callbacks['volunteers']) {
          this.callbacks['volunteers'].forEach(callback => callback(data));
        }
      }
      
      // Traitement pour les notifications de résultat de matrice
      if (messageType === 'matrix_result') {
        if (this.callbacks['matrix_result_' + data.workflow_id]) {
          this.callbacks['matrix_result_' + data.workflow_id].forEach(callback => callback(data));
        }
      }
      
      // Traitement pour les notifications d'avancement de calcul matriciel
      if (messageType === 'matrix_progress') {
        if (this.callbacks['matrix_progress_' + data.workflow_id]) {
          this.callbacks['matrix_progress_' + data.workflow_id].forEach(callback => callback(data));
        }
      }
    };
    
    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      
      if (this.callbacks['error']) {
        this.callbacks['error'].forEach(callback => callback(error));
      }
    };
  }
  
  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.connected = false;
    }
  }
  
  subscribe(eventType, callback) {
    if (!this.callbacks[eventType]) {
      this.callbacks[eventType] = [];
    }
    this.callbacks[eventType].push(callback);
    
    return () => {
      this.callbacks[eventType] = this.callbacks[eventType].filter(cb => cb !== callback);
    };
  }
  
  // S'abonner aux mises à jour d'une tâche spécifique
  subscribeToTaskStatus(taskId, callback) {
    const eventType = 'task_status_' + taskId;
    return this.subscribe(eventType, callback);
  }
  
  // S'abonner aux mises à jour d'un workflow spécifique
  subscribeToWorkflowStatus(workflowId, callback) {
    const eventType = 'workflow_status_' + workflowId;
    return this.subscribe(eventType, callback);
  }
  
  // S'abonner aux mises à jour du Coordinateur
  subscribeToCoordinatorStatus(callback) {
    return this.subscribe('coordinator_status', callback);
  }
  
  // S'abonner aux mises à jour des volontaires
  subscribeToVolunteersUpdates(callback) {
    return this.subscribe('volunteers', callback);
  }
  
  // S'abonner aux résultats de matrice d'un workflow spécifique
  subscribeToMatrixResult(workflowId, callback) {
    const eventType = 'matrix_result_' + workflowId;
    return this.subscribe(eventType, callback);
  }
  
  // S'abonner aux progrès de calcul matriciel d'un workflow spécifique
  subscribeToMatrixProgress(workflowId, callback) {
    const eventType = 'matrix_progress_' + workflowId;
    return this.subscribe(eventType, callback);
  }
  
  send(data) {
    if (this.connected) {
      this.socket.send(JSON.stringify(data));
    } else {
      console.error('Cannot send message: WebSocket not connected');
    }
  }
  
  // Envoyer une demande pour rafraîchir les volontaires
  requestVolunteersRefresh() {
    this.send({
      type: 'request',
      action: 'refresh_volunteers'
    });
  }
  
  // Envoyer une demande pour rafraîchir l'authentification
  requestCoordinatorAuth() {
    this.send({
      type: 'request',
      action: 'authenticate_coordinator'
    });
  }
  
  // Envoyer une demande de statut pour une tâche matricielle
  requestMatrixTaskStatus(taskId) {
    this.send({
      type: 'request',
      action: 'get_matrix_task_status',
      task_id: taskId
    });
  }
  
  // Envoyer une demande de démarrage pour le service de traitement matriciel
  requestStartMatrixHandler() {
    this.send({
      type: 'request',
      action: 'start_matrix_handler'
    });
  }
  
  // Envoyer une demande d'arrêt pour le service de traitement matriciel
  requestStopMatrixHandler() {
    this.send({
      type: 'request',
      action: 'stop_matrix_handler'
    });
  }
  
  // Envoyer une demande de statut pour le service de traitement matriciel
  requestMatrixHandlerStatus() {
    this.send({
      type: 'request',
      action: 'get_matrix_handler_status'
    });
  }
}