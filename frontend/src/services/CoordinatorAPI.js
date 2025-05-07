// src/services/CoordinatorAPI.js

import axios from 'axios';

/**
 * Service pour communiquer avec l'API du coordinateur de calcul distribué
 */
class CoordinatorAPI {
  constructor() {
    this.baseURL = '/api/communication';
    this.token = null;
    this.tokenExpiry = null;
  }

  // Méthode d'authentification auprès du coordinateur
  async authenticate(forceRefresh = false) {
    try {
      const response = await axios.post(`${this.baseURL}/auth/`, {
        force_refresh: forceRefresh
      });
      
      if (response.data && response.data.success) {
        this.token = response.data.token;
        this.tokenExpiry = new Date(response.data.expires_at);
        return {
          success: true,
          token: this.token,
          expires_at: this.tokenExpiry
        };
      }
      
      return {
        success: false,
        error: response.data.error || 'Authentication failed'
      };
    } catch (error) {
      console.error('Error authenticating with coordinator:', error);
      return {
        success: false,
        error: error.message || 'Authentication failed'
      };
    }
  }

  // Méthode pour vérifier si le token est valide
  isTokenValid() {
    if (!this.token || !this.tokenExpiry) {
      return false;
    }
    
    // Ajouter une marge de 60 secondes
    const now = new Date();
    return now < new Date(this.tokenExpiry.getTime() - 60000);
  }

  // Méthode pour obtenir les en-têtes avec authentification
  async getHeaders() {
    if (!this.isTokenValid()) {
      await this.authenticate();
    }
    
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.token}`
    };
  }

  // Méthode générique pour les requêtes
  async request(method, endpoint, data = null, params = null) {
    try {
      const headers = await this.getHeaders();
      
      const config = {
        method,
        url: `${this.baseURL}/${endpoint}`,
        headers
      };
      
      if (data) {
        config.data = data;
      }
      
      if (params) {
        config.params = params;
      }
      
      const response = await axios(config);
      return response.data;
    } catch (error) {
      console.error(`Error in ${method} request to ${endpoint}:`, error);
      
      // Si erreur 401, tenter de se réauthentifier
      if (error.response && error.response.status === 401) {
        this.token = null;
        await this.authenticate(true);
        return this.request(method, endpoint, data, params);
      }
      
      throw error;
    }
  }

  // Méthodes spécifiques aux volontaires
  async getVolunteers(filters = {}) {
    const params = new URLSearchParams();
    
    if (filters.minCpu) {
      params.append('min_cpu', filters.minCpu);
    }
    
    if (filters.minMemory) {
      params.append('min_memory', filters.minMemory);
    }
    
    if (filters.gpu) {
      params.append('gpu', 'true');
    }
    
    if (filters.status) {
      params.append('status', filters.status);
    }
    
    return this.request('get', 'volunteers/', null, params);
  }

  async getVolunteerStatus(volunteerId) {
    return this.request('get', `volunteers/${volunteerId}/status/`);
  }

  // Méthodes spécifiques aux tâches matricielles
  async getMatrixTaskStatus(taskId) {
    return this.request('get', `status/task/${taskId}/`);
  }

  async updateTaskStatus(taskId, status, progress = null) {
    const data = {
      status,
      task_id: taskId
    };
    
    if (progress !== null) {
      data.progress = progress;
    }
    
    return this.request('post', `status/task/${taskId}/`, data);
  }

  // Méthodes spécifiques aux workflows matriciels
  async getWorkflowStatus(workflowId) {
    return this.request('get', `status/workflow/${workflowId}/`);
  }

  async updateWorkflowStatus(workflowId, status, progress = null) {
    const data = {
      status,
      workflow_id: workflowId
    };
    
    if (progress !== null) {
      data.progress = progress;
    }
    
    return this.request('post', `status/workflow/${workflowId}/`, data);
  }

  // Méthodes pour les résultats de tâches
  async getTaskResult(taskId) {
    return this.request('get', `results/task/${taskId}/`);
  }

  async notifyTaskResult(taskId, result) {
    return this.request('post', `results/task/${taskId}/`, {
      task_id: taskId,
      result
    });
  }

  // Méthodes pour le gestionnaire de matrices
  async startMatrixHandler() {
    return this.request('post', 'matrix/start/');
  }

  async stopMatrixHandler() {
    return this.request('post', 'matrix/stop/');
  }

  async getMatrixHandlerStatus() {
    return this.request('get', 'matrix/status/');
  }

  // Méthodes pour les opérations matricielles spécifiques
  async startMatrixAddition(matrixA, matrixB, options = {}) {
    const data = {
      operation: 'addition',
      matrix_a: matrixA,
      matrix_b: matrixB,
      options
    };
    
    return this.request('post', 'matrix/operations/start/', data);
  }

  async startMatrixMultiplication(matrixA, matrixB, options = {}) {
    const data = {
      operation: 'multiplication',
      matrix_a: matrixA,
      matrix_b: matrixB,
      options
    };
    
    return this.request('post', 'matrix/operations/start/', data);
  }

  async getMatrixOperationStatus(operationId) {
    return this.request('get', `matrix/operations/${operationId}/status/`);
  }

  async getMatrixOperationResult(operationId) {
    return this.request('get', `matrix/operations/${operationId}/result/`);
  }
}

// Créer une instance unique du service
const coordinatorAPI = new CoordinatorAPI();

export default coordinatorAPI;