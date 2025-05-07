// src/services/CoordinationService.js

import coordinatorAPI from './CoordinatorAPI';
import WebSocketService from './websocketService';

/**
 * Service centralisé pour coordonner les communications entre le frontend 
 * et les services backend de coordination des tâches matricielles
 */
class CoordinationService {
  constructor() {
    this.webSocketService = new WebSocketService();
    this.coordinatorAPI = coordinatorAPI;
    this.isInitialized = false;
    this.isInitializing = false;
    this.initError = null;
    this.callbacks = {
      coordinatorStatus: [],
      volunteerUpdate: [],
      taskStatus: {},
      workflowStatus: {},
      matrixResult: {},
      matrixProgress: {}
    };
  }

  /**
   * Initialise les services de communication
   * @returns {Promise<boolean>} - Succès de l'initialisation
   */
  async initialize() {
    // Éviter les initialisations multiples
    if (this.isInitialized) return true;
    if (this.isInitializing) {
      console.log('CoordinationService est déjà en cours d\'initialisation');
      // Attendre que l'initialisation en cours se termine
      return new Promise((resolve, reject) => {
        const checkInterval = setInterval(() => {
          if (this.isInitialized) {
            clearInterval(checkInterval);
            resolve(true);
          } else if (this.initError) {
            clearInterval(checkInterval);
            reject(this.initError);
          }
        }, 100);
        
        // Timeout pour éviter d'attendre indéfiniment
        setTimeout(() => {
          clearInterval(checkInterval);
          reject(new Error('Délai d\'attente d\'initialisation dépassé'));
        }, 5000);
      });
    }
    
    this.isInitializing = true;
    this.initError = null;
    
    try {
      console.log('Initialisation du service de coordination...');
      
      // Initialiser la connexion WebSocket
      try {
        console.log('Tentative de connexion WebSocket...');
        this.webSocketService.connect();
        console.log('Connexion WebSocket établie');
      } catch (wsError) {
        console.warn('Erreur lors de la connexion WebSocket:', wsError);
        // Ne pas échouer complètement si WebSocket échoue, mais enregistrer l'erreur
      }
      
      // Configurer les écouteurs WebSocket
      this.setupWebSocketListeners();
      
      // Authentifier auprès du coordinateur (avec un timeout pour éviter les blocages)
      let authResult;
      try {
        console.log('Tentative d\'authentification auprès du coordinateur...');
        const authPromise = this.coordinatorAPI.authenticate();
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Délai d\'authentification dépassé')), 5000);
        });
        
        authResult = await Promise.race([authPromise, timeoutPromise]);
        
        if (!authResult.success) {
          console.warn('Échec de l\'authentification auprès du coordinateur:', authResult.error);
          // Ne pas échouer complètement, enregistrer l'avertissement et continuer
        } else {
          console.log('Authentification réussie auprès du coordinateur');
        }
      } catch (authError) {
        console.warn('Erreur lors de l\'authentification auprès du coordinateur:', authError);
        // Ne pas échouer complètement, enregistrer l'avertissement et continuer
      }
      
      this.isInitialized = true;
      this.isInitializing = false;
      console.log('Service de coordination initialisé avec succès');
      
      // Notifier les abonnés que le service est prêt
      this.notifyCoordinatorStatus({
        status: 'connected',
        authenticated: (authResult && authResult.success) || false,
        token_expiry: (authResult && authResult.expires_at) || null
      });
      
      return true;
    } catch (error) {
      console.error('Erreur critique lors de l\'initialisation du service de coordination:', error);
      this.initError = error;
      this.isInitializing = false;
      
      // Notifier les abonnés de l'échec
      this.notifyCoordinatorStatus({
        status: 'error',
        error: error.message
      });
      
      // Permettre à l'application de continuer même en cas d'erreur
      return false;
    }
  }

  /**
   * Configure les écouteurs pour les événements WebSocket
   */
  setupWebSocketListeners() {
    try {
      // Événement de connexion
      this.webSocketService.subscribe('connect', () => {
        console.log('WebSocket connecté');
        this.notifyCoordinatorStatus({ status: 'connected' });
      });
      
      // Événement de déconnexion
      this.webSocketService.subscribe('disconnect', (event) => {
        console.log('WebSocket déconnecté', event);
        this.notifyCoordinatorStatus({ status: 'disconnected' });
      });
      
      // Erreur WebSocket
      this.webSocketService.subscribe('error', (error) => {
        console.error('Erreur WebSocket:', error);
        this.notifyCoordinatorStatus({ status: 'error', error });
      });
      
      // Mise à jour du coordinateur
      this.webSocketService.subscribe('coordinator_status', (data) => {
        this.notifyCoordinatorStatus(data);
      });
      
      // Mise à jour des volontaires
      this.webSocketService.subscribe('volunteers', (data) => {
        this.notifyVolunteerUpdate(data);
      });
      
      // Événements génériques
      this.webSocketService.subscribe('message', (data) => {
        console.log('Message WebSocket reçu:', data);
        
        // Acheminer les messages vers les gestionnaires appropriés
        switch (data.type) {
          case 'task_update':
            this.notifyTaskStatus(data.task_id, data);
            break;
            
          case 'workflow_update':
            this.notifyWorkflowStatus(data.workflow_id, data);
            break;
            
          case 'matrix_result':
            this.notifyMatrixResult(data.workflow_id, data);
            break;
            
          case 'matrix_progress':
            this.notifyMatrixProgress(data.workflow_id, data);
            break;
            
          default:
            // Traiter les messages de type inconnu
            console.log(`Message avec type inconnu: ${data.type}`, data);
            break;
        }
      });
    } catch (error) {
      console.error('Erreur lors de la configuration des écouteurs WebSocket:', error);
      // Ne pas propager cette erreur, pour éviter de bloquer l'initialisation
    }
  }

  /**
   * Notifie les abonnés aux mises à jour de statut du coordinateur
   * @param {Object} status - Les informations de statut
   */
  notifyCoordinatorStatus(status) {
    try {
      this.callbacks.coordinatorStatus.forEach(callback => {
        try {
          callback(status);
        } catch (error) {
          console.error('Erreur dans le callback de statut du coordinateur:', error);
        }
      });
    } catch (error) {
      console.error('Erreur lors de la notification des abonnés au statut du coordinateur:', error);
    }
  }

  /**
   * Notifie les abonnés aux mises à jour des volontaires
   * @param {Object} data - Les données de mise à jour
   */
  notifyVolunteerUpdate(data) {
    try {
      this.callbacks.volunteerUpdate.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Erreur dans le callback de mise à jour des volontaires:', error);
        }
      });
    } catch (error) {
      console.error('Erreur lors de la notification des abonnés aux mises à jour des volontaires:', error);
    }
  }

  /**
   * Notifie les abonnés aux mises à jour de statut de tâche
   * @param {string} taskId - L'identifiant de la tâche
   * @param {Object} status - Les informations de statut
   */
  notifyTaskStatus(taskId, status) {
    try {
      if (this.callbacks.taskStatus[taskId]) {
        this.callbacks.taskStatus[taskId].forEach(callback => {
          try {
            callback(status);
          } catch (error) {
            console.error(`Erreur dans le callback de statut de tâche pour la tâche ${taskId}:`, error);
          }
        });
      }
    } catch (error) {
      console.error(`Erreur lors de la notification des abonnés au statut de la tâche ${taskId}:`, error);
    }
  }

  /**
   * Notifie les abonnés aux mises à jour de statut de workflow
   * @param {string} workflowId - L'identifiant du workflow
   * @param {Object} status - Les informations de statut
   */
  notifyWorkflowStatus(workflowId, status) {
    try {
      if (this.callbacks.workflowStatus[workflowId]) {
        this.callbacks.workflowStatus[workflowId].forEach(callback => {
          try {
            callback(status);
          } catch (error) {
            console.error(`Erreur dans le callback de statut de workflow pour le workflow ${workflowId}:`, error);
          }
        });
      }
    } catch (error) {
      console.error(`Erreur lors de la notification des abonnés au statut du workflow ${workflowId}:`, error);
    }
  }

  /**
   * Notifie les abonnés aux résultats de matrices
   * @param {string} workflowId - L'identifiant du workflow
   * @param {Object} result - Les résultats de la matrice
   */
  notifyMatrixResult(workflowId, result) {
    try {
      if (this.callbacks.matrixResult[workflowId]) {
        this.callbacks.matrixResult[workflowId].forEach(callback => {
          try {
            callback(result);
          } catch (error) {
            console.error(`Erreur dans le callback de résultats de matrice pour le workflow ${workflowId}:`, error);
          }
        });
      }
    } catch (error) {
      console.error(`Erreur lors de la notification des abonnés aux résultats de matrice pour le workflow ${workflowId}:`, error);
    }
  }

  /**
   * Notifie les abonnés à la progression du calcul matriciel
   * @param {string} workflowId - L'identifiant du workflow
   * @param {Object} progress - Les informations de progression
   */
  notifyMatrixProgress(workflowId, progress) {
    try {
      if (this.callbacks.matrixProgress[workflowId]) {
        this.callbacks.matrixProgress[workflowId].forEach(callback => {
          try {
            callback(progress);
          } catch (error) {
            console.error(`Erreur dans le callback de progression de matrice pour le workflow ${workflowId}:`, error);
          }
        });
      }
    } catch (error) {
      console.error(`Erreur lors de la notification des abonnés à la progression de matrice pour le workflow ${workflowId}:`, error);
    }
  }

  /**
   * S'abonne aux mises à jour de statut du coordinateur
   * @param {Function} callback - Fonction de rappel à appeler lors des mises à jour
   * @returns {Function} - Fonction pour se désabonner
   */
  subscribeToCoordinatorStatus(callback) {
    this.callbacks.coordinatorStatus.push(callback);
    
    // Renvoyer une fonction de désabonnement
    return () => {
      this.callbacks.coordinatorStatus = this.callbacks.coordinatorStatus.filter(cb => cb !== callback);
    };
  }

  /**
   * S'abonne aux mises à jour des volontaires
   * @param {Function} callback - Fonction de rappel à appeler lors des mises à jour
   * @returns {Function} - Fonction pour se désabonner
   */
  subscribeToVolunteers(callback) {
    this.callbacks.volunteerUpdate.push(callback);
    
    // S'abonner via WebSocket si disponible
    if (this.webSocketService && this.webSocketService.connected) {
      try {
        this.webSocketService.subscribeToVolunteersUpdates(callback);
      } catch (error) {
        console.warn('Impossible de s\'abonner aux mises à jour des volontaires via WebSocket:', error);
      }
    }
    
    // Renvoyer une fonction de désabonnement
    return () => {
      this.callbacks.volunteerUpdate = this.callbacks.volunteerUpdate.filter(cb => cb !== callback);
    };
  }

  /**
   * S'abonne aux mises à jour des volontaires
   * @param {Function} callback - Fonction de rappel à appeler lors des mises à jour
   * @returns {Function} - Fonction pour se désabonner
   */
  subscribeToVolunteersUpdate(callback) {
    this.callbacks.volunteerUpdate.push(callback);
    
    // Renvoyer une fonction de désabonnement
    return () => {
      this.callbacks.volunteerUpdate = this.callbacks.volunteerUpdate.filter(cb => cb !== callback);
    };
  }

  /**
   * S'abonne aux mises à jour de statut d'une tâche spécifique
   * @param {string} taskId - L'identifiant de la tâche
   * @param {Function} callback - Fonction de rappel à appeler lors des mises à jour
   * @returns {Function} - Fonction pour se désabonner
   */
  subscribeToTaskStatus(taskId, callback) {
    if (!this.callbacks.taskStatus[taskId]) {
      this.callbacks.taskStatus[taskId] = [];
    }
    
    this.callbacks.taskStatus[taskId].push(callback);
    
    // Renvoyer une fonction de désabonnement
    return () => {
      this.callbacks.taskStatus[taskId] = this.callbacks.taskStatus[taskId].filter(cb => cb !== callback);
    };
  }

  /**
   * S'abonne aux mises à jour de statut d'un workflow spécifique
   * @param {string} workflowId - L'identifiant du workflow
   * @param {Function} callback - Fonction de rappel à appeler lors des mises à jour
   * @returns {Function} - Fonction pour se désabonner
   */
  subscribeToWorkflowStatus(workflowId, callback) {
    if (!this.callbacks.workflowStatus[workflowId]) {
      this.callbacks.workflowStatus[workflowId] = [];
    }
    
    this.callbacks.workflowStatus[workflowId].push(callback);
    
    // S'abonner via WebSocket si disponible
    if (this.webSocketService && this.webSocketService.connected) {
      try {
        this.webSocketService.subscribeToWorkflowStatus(workflowId, callback);
      } catch (error) {
        console.warn(`Impossible de s'abonner au statut du workflow ${workflowId} via WebSocket:`, error);
      }
    }
    
    // Renvoyer une fonction de désabonnement
    return () => {
      this.callbacks.workflowStatus[workflowId] = this.callbacks.workflowStatus[workflowId].filter(cb => cb !== callback);
    };
  }

  /**
   * S'abonne aux résultats de matrices pour un workflow spécifique
   * @param {string} workflowId - L'identifiant du workflow
   * @param {Function} callback - Fonction de rappel à appeler lors de la réception des résultats
   * @returns {Function} - Fonction pour se désabonner
   */
  subscribeToMatrixResult(workflowId, callback) {
    if (!this.callbacks.matrixResult[workflowId]) {
      this.callbacks.matrixResult[workflowId] = [];
    }
    
    this.callbacks.matrixResult[workflowId].push(callback);
    
    // S'abonner via WebSocket si disponible
    if (this.webSocketService && this.webSocketService.connected) {
      try {
        this.webSocketService.subscribeToMatrixResult(workflowId, callback);
      } catch (error) {
        console.warn(`Impossible de s'abonner aux résultats de matrice pour le workflow ${workflowId} via WebSocket:`, error);
      }
    }
    
    // Renvoyer une fonction de désabonnement
    return () => {
      this.callbacks.matrixResult[workflowId] = this.callbacks.matrixResult[workflowId].filter(cb => cb !== callback);
    };
  }

  /**
   * S'abonne à la progression du calcul matriciel pour un workflow spécifique
   * @param {string} workflowId - L'identifiant du workflow
   * @param {Function} callback - Fonction de rappel à appeler lors des mises à jour de progression
   * @returns {Function} - Fonction pour se désabonner
   */
  subscribeToMatrixProgress(workflowId, callback) {
    if (!this.callbacks.matrixProgress[workflowId]) {
      this.callbacks.matrixProgress[workflowId] = [];
    }
    
    this.callbacks.matrixProgress[workflowId].push(callback);
    
    // S'abonner via WebSocket si disponible
    if (this.webSocketService && this.webSocketService.connected) {
      try {
        this.webSocketService.subscribeToMatrixProgress(workflowId, callback);
      } catch (error) {
        console.warn(`Impossible de s'abonner à la progression de matrice pour le workflow ${workflowId} via WebSocket:`, error);
      }
    }
    
    // Renvoyer une fonction de désabonnement
    return () => {
      this.callbacks.matrixProgress[workflowId] = this.callbacks.matrixProgress[workflowId].filter(cb => cb !== callback);
    };
  }

  /**
   * Déconnecte le service de coordination
   */
  disconnect() {
    // Vérifier si le service est initialisé
    if (!this.isInitialized) {
      console.log('Le service de coordination n\'est pas initialisé, aucune déconnexion nécessaire');
      return;
    }
    
    // Déconnecter WebSocket si disponible
    if (this.webSocketService) {
      try {
        this.webSocketService.disconnect();
      } catch (error) {
        console.warn('Erreur lors de la déconnexion WebSocket:', error);
      }
    }
    
    this.isInitialized = false;
    
    // Réinitialiser les callbacks
    this.callbacks = {
      coordinatorStatus: [],
      volunteerUpdate: [],
      taskStatus: {},
      workflowStatus: {},
      matrixResult: {},
      matrixProgress: {}
    };
    
    console.log('Service de coordination déconnecté');
  }

  /**
   * Demande le statut actuel du coordinateur
   * @returns {Promise<Object>} - Les informations de statut du coordinateur
   */
  async getCoordinatorStatus() {
    try {
      // Si non initialisé, initialiser le service
      if (!this.isInitialized) {
        await this.initialize();
      }
      
      // Utiliser l'API du coordinateur si disponible
      let isTokenValid = false;
      try {
        isTokenValid = this.coordinatorAPI.isTokenValid();
      } catch (error) {
        console.warn('Erreur lors de la vérification de la validité du token:', error);
      }
      
      // Obtenir les informations d'état via le WebSocket si disponible
      if (this.webSocketService && this.webSocketService.connected) {
        try {
          this.webSocketService.requestCoordinatorAuth();
        } catch (error) {
          console.warn('Erreur lors de la demande de statut du coordinateur via WebSocket:', error);
        }
      }
      
      return {
        connected: this.webSocketService ? this.webSocketService.connected : false,
        authenticated: isTokenValid,
        token_expiry: this.coordinatorAPI ? this.coordinatorAPI.tokenExpiry : null
      };
    } catch (error) {
      console.error('Erreur lors de la récupération du statut du coordinateur:', error);
      return {
        connected: false,
        authenticated: false,
        error: error.message
      };
    }
  }

  /**
   * Démarre le gestionnaire de matrices
   * @returns {Promise<Object>} - Résultat de l'opération
   */
  async startMatrixHandler() {
    try {
      // Si non initialisé, initialiser le service
      if (!this.isInitialized) {
        await this.initialize();
      }
      
      // Utiliser le WebSocket si disponible
      if (this.webSocketService && this.webSocketService.connected) {
        try {
          this.webSocketService.requestStartMatrixHandler();
        } catch (error) {
          console.warn('Erreur lors de la demande de démarrage du gestionnaire de matrices via WebSocket:', error);
        }
      }
      
      // Utiliser l'API REST
      let result = { success: false };
      try {
        result = await this.coordinatorAPI.startMatrixHandler();
      } catch (error) {
        console.error('Erreur lors du démarrage du gestionnaire de matrices via API REST:', error);
        throw error;
      }
      
      return result;
    } catch (error) {
      console.error('Erreur lors du démarrage du gestionnaire de matrices:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Arrête le gestionnaire de matrices
   * @returns {Promise<Object>} - Résultat de l'opération
   */
  async stopMatrixHandler() {
    try {
      // Si non initialisé, initialiser le service
      if (!this.isInitialized) {
        await this.initialize();
      }
      
      // Utiliser le WebSocket si disponible
      if (this.webSocketService && this.webSocketService.connected) {
        try {
          this.webSocketService.requestStopMatrixHandler();
        } catch (error) {
          console.warn('Erreur lors de la demande d\'arrêt du gestionnaire de matrices via WebSocket:', error);
        }
      }
      
      // Utiliser l'API REST
      let result = { success: false };
      try {
        result = await this.coordinatorAPI.stopMatrixHandler();
      } catch (error) {
        console.error('Erreur lors de l\'arrêt du gestionnaire de matrices via API REST:', error);
        throw error;
      }
      
      return result;
    } catch (error) {
      console.error('Erreur lors de l\'arrêt du gestionnaire de matrices:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Obtient le statut du gestionnaire de matrices
   * @returns {Promise<Object>} - Statut du gestionnaire de matrices
   */
  async getMatrixHandlerStatus() {
    try {
      // Si non initialisé, initialiser le service
      if (!this.isInitialized) {
        await this.initialize();
      }
      
      // Utiliser le WebSocket si disponible
      if (this.webSocketService && this.webSocketService.connected) {
        try {
          this.webSocketService.requestMatrixHandlerStatus();
        } catch (error) {
          console.warn('Erreur lors de la demande de statut du gestionnaire de matrices via WebSocket:', error);
        }
      }
      
      // Utiliser l'API REST
      let result = { success: false, status: 'unknown' };
      try {
        result = await this.coordinatorAPI.getMatrixHandlerStatus();
      } catch (error) {
        console.error('Erreur lors de la récupération du statut du gestionnaire de matrices via API REST:', error);
        throw error;
      }
      
      return result;
    } catch (error) {
      console.error('Erreur lors de la récupération du statut du gestionnaire de matrices:', error);
      return {
        success: false,
        status: 'unknown',
        error: error.message
      };
    }
  }

  /**
   * Obtient le statut d'un workflow matriciel
   * @param {string} workflowId - L'identifiant du workflow
   * @returns {Promise<Object>} - Statut du workflow
   */
  async getWorkflowStatus(workflowId) {
    try {
      // Si non initialisé, initialiser le service
      if (!this.isInitialized) {
        await this.initialize();
      }
      
      // Utiliser l'API REST
      let result = { success: false, workflow_id: workflowId, status: 'unknown' };
      try {
        result = await this.coordinatorAPI.getWorkflowStatus(workflowId);
      } catch (error) {
        console.error(`Erreur lors de la récupération du statut du workflow ${workflowId} via API REST:`, error);
        throw error;
      }
      
      return result;
    } catch (error) {
      console.error(`Erreur lors de la récupération du statut du workflow ${workflowId}:`, error);
      return {
        success: false,
        workflow_id: workflowId,
        status: 'unknown',
        error: error.message
      };
    }
  }

  /**
   * Obtient le statut d'une tâche matricielle
   * @param {string} taskId - L'identifiant de la tâche
   * @returns {Promise<Object>} - Statut de la tâche
   */
  async getTaskStatus(taskId) {
    try {
      // Si non initialisé, initialiser le service
      if (!this.isInitialized) {
        await this.initialize();
      }
      
      // Utiliser le WebSocket si disponible
      if (this.webSocketService && this.webSocketService.connected) {
        try {
          this.webSocketService.requestMatrixTaskStatus(taskId);
        } catch (error) {
          console.warn(`Erreur lors de la demande de statut de la tâche ${taskId} via WebSocket:`, error);
        }
      }
      
      // Utiliser l'API REST
      let result = { success: false, task_id: taskId, status: 'unknown' };
      try {
        result = await this.coordinatorAPI.getMatrixTaskStatus(taskId);
      } catch (error) {
        console.error(`Erreur lors de la récupération du statut de la tâche ${taskId} via API REST:`, error);
        throw error;
      }
      
      return result;
    } catch (error) {
      console.error(`Erreur lors de la récupération du statut de la tâche ${taskId}:`, error);
      return {
        success: false,
        task_id: taskId,
        status: 'unknown',
        error: error.message
      };
    }
  }
}

// Créer et exporter une instance unique du service
const coordinationService = new CoordinationService();

export default coordinationService;