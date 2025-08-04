// lib/api.ts
import axios from 'axios';

// Configuration de l'instance API avec la bonne URL de base
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Intercepteur pour ajouter le token d'authentification
api.interceptors.request.use(
  (config) => {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers.Authorization = `Token ${token}`;
      }
    }
    
    // Log pour débogage (sans afficher les mots de passe)
    console.log(`[API] Requête ${config.method?.toUpperCase()} ${config.url}:`, {
      headers: { ...config.headers, Authorization: config.headers.Authorization ? 'Token ***' : undefined },
      data: config.data ? { 
        ...config.data, 
        password: config.data.password ? '********' : undefined,
        password2: config.data.password2 ? '********' : undefined
      } : undefined
    });
    
    return config;
  },
  (error) => {
    console.error('[API] Erreur de requête:', error);
    return Promise.reject(error);
  }
);

// Intercepteur pour gérer les réponses et erreurs
api.interceptors.response.use(
  (response) => {
    console.log(`[API] Réponse ${response.config.url}:`, {
      status: response.status,
      statusText: response.statusText,
      data: response.data
    });
    return response;
  },
  (error) => {
    const errorInfo = {
      url: error.config?.url || 'inconnu',
      method: error.config?.method?.toUpperCase() || 'INCONNU',
      status: error.response?.status || 'aucun status',
      statusText: error.response?.statusText || 'aucun status text',
      data: error.response?.data || 'aucune donnée',
      message: error.message
    };
    
    console.error(`[API] Erreur ${errorInfo.method} ${errorInfo.url}:`, errorInfo);
    
    // Gestion spéciale des erreurs réseau
    if (!error.response) {
      console.error('[API] Erreur réseau - Le serveur pourrait être arrêté ou inaccessible');
    }
    
    return Promise.reject(error);
  }
);

// Fonction utilitaire pour extraire les messages d'erreur
const extractErrorMessage = (error: any): string => {
  // Erreur réseau
  if (!error.response) {
    return 'Impossible de contacter le serveur. Vérifiez que le serveur Django est démarré.';
  }
  
  const { status, data } = error.response;
  
  // Erreurs serveur (5xx)
  if (status >= 500) {
    console.error('Détails erreur serveur:', {
      status,
      statusText: error.response.statusText,
      data,
      headers: error.response.headers
    });
    return `Erreur serveur (${status}). Vérifiez les logs du serveur Django.`;
  }
  
  // Erreur 404
  if (status === 404) {
    return 'Endpoint non trouvé. Vérifiez la configuration des URLs Django.';
  }
  
  // Erreurs client (4xx)
  if (status >= 400) {
    if (typeof data === 'string') {
      return data;
    }
    
    if (data && typeof data === 'object') {
      // Formats d'erreur Django REST Framework
      if (data.detail) return data.detail;
      if (data.error) return data.error;
      if (data.message) return data.message;
      
      // Erreurs de champs spécifiques
      if (data.email) {
        const emailError = Array.isArray(data.email) ? data.email[0] : data.email;
        return `Email: ${emailError}`;
      }
      if (data.username) {
        const usernameError = Array.isArray(data.username) ? data.username[0] : data.username;
        return `Nom d'utilisateur: ${usernameError}`;
      }
      if (data.password) {
        const passwordError = Array.isArray(data.password) ? data.password[0] : data.password;
        return `Mot de passe: ${passwordError}`;
      }
      if (data.non_field_errors) {
        const nonFieldError = Array.isArray(data.non_field_errors) ? data.non_field_errors[0] : data.non_field_errors;
        return nonFieldError;
      }
      
      // Première erreur trouvée
      const firstKey = Object.keys(data)[0];
      if (firstKey && data[firstKey]) {
        const errorValue = Array.isArray(data[firstKey]) ? data[firstKey][0] : data[firstKey];
        return `${firstKey}: ${errorValue}`;
      }
    }
  }
  
  return 'Une erreur inattendue s\'est produite.';
};

// Service d'authentification avec les bons endpoints
export const authService = {
  // Test de connectivité
  testConnection: async () => {
    try {
      console.log('[API] Test de connexion au serveur...');
      // Tentative de connexion à un endpoint connu
      const response = await api.options('/workflows/auth/login/');
      console.log('[API] Test de connexion réussi');
      return { success: true, message: 'Serveur accessible' };
    } catch (error) {
      console.error('[API] Test de connexion échoué:', error);
      return { 
        success: false, 
        error: extractErrorMessage(error),
        details: error
      };
    }
  },

  // Inscription
  register: async (userData: {
    first_name: string;
    last_name: string;
    email: string;
    password: string;
    password2: string;
  }) => {
    try {
      console.log('[AUTH] Début de l\'inscription...');
      
      // Validation côté client
      if (!userData.first_name || !userData.last_name || !userData.email || !userData.password || !userData.password2) {
        throw new Error('Tous les champs sont requis');
      }
      
      if (userData.password !== userData.password2) {
        throw new Error('Les mots de passe ne correspondent pas');
      }
      
      // Validation email basique
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(userData.email)) {
        throw new Error('Format d\'email invalide');
      }
      
      // Envoi de la requête d'inscription
      const response = await api.post('/workflows/auth/register/', userData);
      
      // Gestion de la réponse
      if (response.data && response.data.token) {
        localStorage.setItem('token', response.data.token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        console.log('[AUTH] Inscription réussie');
      } else {
        console.warn('[AUTH] Réponse d\'inscription sans token:', response.data);
      }
      
      return response.data;
    } catch (error: any) {
      console.error('[AUTH] Erreur d\'inscription:', error);
      throw new Error(extractErrorMessage(error));
    }
  },

  // Connexion
  login: async (credentials: { email: string; password: string }) => {
    try {
      console.log('[AUTH] Début de la connexion...');
      
      // Validation des données
      if (!credentials.email || !credentials.password) {
        throw new Error('Email et mot de passe requis');
      }
      
      // Validation du format email
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(credentials.email)) {
        throw new Error('Format d\'email invalide');
      }
      
      // Envoi de la requête de connexion avec l'URL correcte
      const response = await api.post('/workflows/auth/login/', credentials);
      
      // Gestion de la réponse
      if (response.data && response.data.token) {
        localStorage.setItem('token', response.data.token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        console.log('[AUTH] Connexion réussie');
      } else {
        console.warn('[AUTH] Réponse de connexion sans token:', response.data);
        throw new Error('Réponse du serveur invalide - token manquant');
      }
      
      return response.data;
    } catch (error: any) {
      console.error('[AUTH] Erreur de connexion:', error);
      throw new Error(extractErrorMessage(error));
    }
  },

  // Déconnexion
  logout: async () => {
    try {
      console.log('[AUTH] Début de la déconnexion...');
      await api.post('/workflows/auth/logout/');
      console.log('[AUTH] Déconnexion réussie');
    } catch (error) {
      console.error('[AUTH] Erreur de déconnexion (non critique):', error);
    } finally {
      // Nettoyage du stockage local dans tous les cas
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      console.log('[AUTH] Stockage local nettoyé');
    }
  },

  // Vérification de l'authentification
  isAuthenticated: () => {
    if (typeof window === 'undefined') return false;
    
    const token = localStorage.getItem('token');
    const user = localStorage.getItem('user');
    
    if (!token || !user) {
      return false;
    }
    
    // Vérification de la validité des données utilisateur
    try {
      JSON.parse(user);
      return true;
    } catch (error) {
      console.error('[AUTH] Données utilisateur invalides dans localStorage:', error);
      localStorage.removeItem('user');
      localStorage.removeItem('token');
      return false;
    }
  },

  // Récupération de l'utilisateur actuel
  getCurrentUser: () => {
    if (typeof window === 'undefined') return null;
    
    const userStr = localStorage.getItem('user');
    if (!userStr) return null;
    
    try {
      return JSON.parse(userStr);
    } catch (error) {
      console.error('[AUTH] Erreur lors de l\'analyse des données utilisateur:', error);
      localStorage.removeItem('user');
      localStorage.removeItem('token');
      return null;
    }
  },

  // Rafraîchissement du token (si votre backend le supporte)
  refreshToken: async () => {
    try {
      const response = await api.post('/workflows/auth/refresh/');
      if (response.data && response.data.token) {
        localStorage.setItem('token', response.data.token);
        return response.data;
      }
    } catch (error) {
      console.error('[AUTH] Échec du rafraîchissement du token:', error);
      // Nettoyage du token invalide
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      throw error;
    }
  }
};

// Services pour les workflows
export const workflowService = {
  // Récupérer tous les workflows
  getWorkflows: async () => {
    try {
      const response = await api.get('/workflows/');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des workflows' };
      }
    }
  },

  // Récupérer un workflow par ID
  getWorkflow: async (id: string) => {
    try {
      const response = await api.get(`/workflows/${id}/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération du workflow' };
      }
    }
  },

  // Créer un workflow
  createWorkflow: async (workflowData: any) => {
    try {
      const response = await api.post('/workflows/', workflowData);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la création du workflow' };
      }
    }
  },

  // Mettre à jour un workflow
  updateWorkflow: async (id: string, workflowData: any) => {
    try {
      const response = await api.put(`/workflows/${id}/`, workflowData);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la mise à jour du workflow' };
      }
    }
  },

  // Supprimer un workflow
  deleteWorkflow: async (id: string) => {
    try {
      const response = await api.delete(`/workflows/${id}/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la suppression du workflow' };
      }
    }
  },

  // Soumettre un workflow
  submitWorkflow: async (id: string) => {
    try {
      const response = await api.post(`/workflows/${id}/submit/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la soumission du workflow' };
      }
    }
  },

  // Récupérer les tâches d'un workflow
  getWorkflowTasks: async (id: string) => {
    try {
      const response = await api.get(`/workflows/${id}/tasks/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des tâches du workflow' };
      }
    }
  }
};

// Gestion des tâches
export const taskService = {
  // Récupérer toutes les tâches
  getTasks: async () => {
    try {
      const response = await api.get('/tasks/');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des tâches' };
      }
    }
  },

  // Récupérer une tâche par ID
  getTask: async (id: string) => {
    try {
      const response = await api.get(`/tasks/${id}/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération de la tâche' };
      }
    }
  },

  // Récupérer les tâches d'un workflow
  getWorkflowTasks: async (workflowId: string) => {
    try {
      const response = await api.get(`/tasks/by_workflow/?workflow_id=${workflowId}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des tâches du workflow' };
      }
    }
  },

  // Créer une tâche
  createTask: async (taskData: any) => {
    try {
      const response = await api.post('/tasks/', taskData);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la création de la tâche' };
      }
    }
  },

  // Mettre à jour une tâche
  updateTask: async (id: string, taskData: any) => {
    try {
      const response = await api.put(`/tasks/${id}/`, taskData);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la mise à jour de la tâche' };
      }
    }
  },

  // Supprimer une tâche
  deleteTask: async (id: string) => {
    try {
      const response = await api.delete(`/tasks/${id}/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la suppression de la tâche' };
      }
    }
  },

  // Assigner une tâche à un volontaire
  assignTask: async (taskId: string, volunteerId: string) => {
    try {
      const response = await api.post(`/tasks/${taskId}/assign/`, { volunteer_id: volunteerId });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de l\'assignation de la tâche' };
      }
    }
  },

  // Récupérer les volontaires assignés à une tâche
  getTaskVolunteers: async (taskId: string) => {
    try {
      const response = await api.get(`/tasks/${taskId}/volunteers/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des volontaires de la tâche' };
      }
    }
  },

  // Récupérer les tâches par statut
  getTasksByStatus: async (status: string) => {
    try {
      const response = await api.get(`/tasks/by_status/?status=${status}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des tâches par statut' };
      }
    }
  },

  // Démarrer une tâche
  startTask: async (taskId: string) => {
    try {
      const response = await api.post(`/tasks/${taskId}/start/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors du démarrage de la tâche' };
      }
    }
  },

  // Terminer une tâche
  completeTask: async (taskId: string) => {
    try {
      const response = await api.post(`/tasks/${taskId}/complete/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la complétion de la tâche' };
      }
    }
  },

  // Marquer une tâche comme échouée
  failTask: async (taskId: string, errorMessage: string = '') => {
    try {
      const response = await api.post(`/tasks/${taskId}/fail/`, { error: errorMessage });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors du marquage de la tâche comme échouée' };
      }
    }
  },

  // Mettre à jour la progression d'une tâche
  updateTaskProgress: async (taskId: string, progress: number) => {
    try {
      const response = await api.post(`/tasks/${taskId}/update_progress/`, { progress });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la mise à jour de la progression' };
      }
    }
  }
};

// Gestion des volontaires
export const volunteerService = {
  // Récupérer tous les volontaires
  getVolunteers: async () => {
    try {
      const response = await api.get('/volunteers/');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des volontaires' };
      }
    }
  },

  // Récupérer un volontaire par ID
  getVolunteer: async (id: string) => {
    try {
      const response = await api.get(`/volunteers/${id}/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération du volontaire' };
      }
    }
  },

  // Récupérer les volontaires par workflow
  getWorkflowVolunteers: async (workflowId: string) => {
    try {
      const response = await api.get(`/volunteers/by_workflow/?workflow_id=${workflowId}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des volontaires du workflow' };
      }
    }
  },

  // Récupérer les volontaires par statut
  getVolunteersByStatus: async (status: string) => {
    try {
      const response = await api.get(`/volunteers/by_status/?status=${status}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des volontaires par statut' };
      }
    }
  },

  // Récupérer les tâches assignées à un volontaire
  getVolunteerTasks: async (volunteerId: string) => {
    try {
      const response = await api.get(`/volunteers/${volunteerId}/tasks/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des tâches du volontaire' };
      }
    }
  },

  // Assigner une tâche à un volontaire
  assignTask: async (volunteerId: string, taskId: string) => {
    try {
      const response = await api.post(`/volunteers/${volunteerId}/assign_task/`, { task_id: taskId });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de l\'assignation de la tâche' };
      }
    }
  },

  // Mettre à jour le statut d'un volontaire
  updateVolunteerStatus: async (volunteerId: string, status: string) => {
    try {
      const response = await api.patch(`/volunteers/${volunteerId}/`, { status });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la mise à jour du statut du volontaire' };
      }
    }
  },

  // Mettre à jour la disponibilité d'un volontaire
  updateVolunteerAvailability: async (volunteerId: string, available: boolean) => {
    try {
      const response = await api.patch(`/volunteers/${volunteerId}/`, { available });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la mise à jour de la disponibilité du volontaire' };
      }
    }
  }
};

// Gestion des assignations de tâches aux volontaires
export const volunteerTaskService = {
  // Récupérer toutes les assignations
  getVolunteerTasks: async () => {
    try {
      const response = await api.get('/volunteers/tasks/');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des assignations' };
      }
    }
  },

  // Récupérer une assignation par ID
  getVolunteerTask: async (id: string) => {
    try {
      const response = await api.get(`/volunteers/tasks/${id}/`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération de l\'assignation' };
      }
    }
  },

  // Récupérer les assignations par tâche
  getTaskAssignments: async (taskId: string) => {
    try {
      const response = await api.get(`/volunteers/tasks/by_task/?task_id=${taskId}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des assignations par tâche' };
      }
    }
  },

  // Récupérer les assignations par volontaire
  getVolunteerAssignments: async (volunteerId: string) => {
    try {
      const response = await api.get(`/volunteers/tasks/by_volunteer/?volunteer_id=${volunteerId}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la récupération des assignations par volontaire' };
      }
    }
  },

  // Mettre à jour la progression d'une assignation
  updateProgress: async (volunteerTaskId: string, progress: number) => {
    try {
      const response = await api.post(`/volunteers/tasks/${volunteerTaskId}/update_progress/`, { progress });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors de la mise à jour de la progression' };
      }
    }
  },

  // Marquer une assignation comme terminée
  completeTask: async (volunteerTaskId: string, result: any = null) => {
    try {
      const response = await api.post(`/volunteers/tasks/${volunteerTaskId}/complete/`, { result });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors du marquage de l\'assignation comme terminée' };
      }
    }
  },

  // Marquer une assignation comme échouée
  failTask: async (volunteerTaskId: string, error: string) => {
    try {
      const response = await api.post(`/volunteers/tasks/${volunteerTaskId}/fail/`, { error });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw error.response.data;
      } else {
        throw { error: 'Une erreur est survenue lors du marquage de l\'assignation comme échouée' };
      }
    }
  }
};


// Service WebSocket
export const websocketService = {
  // URL de base WebSocket
  WS_BASE_URL: 'ws://127.0.0.1:8000/ws/workflows/',

  // Créer une connexion WebSocket avec authentification
  connect: (token: string) => {
    const wsUrl = `${websocketService.WS_BASE_URL}?token=${encodeURIComponent(token)}`;
    return new WebSocket(wsUrl);
  },

  // Vérifier l'état de la connexion WebSocket
  isConnected: (ws: WebSocket | null): boolean => {
    return ws !== null && ws.readyState === WebSocket.OPEN;
  },

  // Envoyer un message via WebSocket
  send: (ws: WebSocket, message: any) => {
    if (websocketService.isConnected(ws)) {
      ws.send(JSON.stringify(message));
      return true;
    }
    console.warn('WebSocket non connecté, impossible d\'envoyer le message');
    return false;
  },

  // S'abonner aux mises à jour d'un workflow
  subscribeToWorkflow: (ws: WebSocket, workflowId: string) => {
    return websocketService.send(ws, {
      type: 'subscribe_workflow',
      workflow_id: workflowId
    });
  },

  // S'abonner aux mises à jour d'une tâche
  subscribeToTask: (ws: WebSocket, taskId: string) => {
    return websocketService.send(ws, {
      type: 'subscribe_task',
      task_id: taskId
    });
  },

  // S'abonner aux mises à jour d'un volontaire
  subscribeToVolunteer: (ws: WebSocket, volunteerId: string) => {
    return websocketService.send(ws, {
      type: 'subscribe_volunteer',
      volunteer_id: volunteerId
    });
  },

  // Envoyer un ping
  ping: (ws: WebSocket) => {
    return websocketService.send(ws, { type: 'ping' });
  },

  // Gestionnaire de messages générique
  handleMessage: (event: MessageEvent, callbacks: {
    onWorkflowUpdate?: (data: any) => void;
    onWorkflowStatusChange?: (data: any) => void;
    onTaskUpdate?: (data: any) => void;
    onTaskProgress?: (data: any) => void;
    onVolunteerUpdate?: (data: any) => void;
    onVolunteerStatus?: (data: any) => void;
    onError?: (data: any) => void;
  }) => {
    try {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'workflow_update':
          callbacks.onWorkflowUpdate?.(data);
          break;
        case 'workflow_status_change':
          callbacks.onWorkflowStatusChange?.(data);
          break;
        case 'task_update':
          callbacks.onTaskUpdate?.(data);
          break;
        case 'task_progress':
          callbacks.onTaskProgress?.(data);
          break;
        case 'volunteer_update':
          callbacks.onVolunteerUpdate?.(data);
          break;
        case 'volunteer_status':
          callbacks.onVolunteerStatus?.(data);
          break;
        case 'error':
          callbacks.onError?.(data);
          break;
        default:
          console.log('Message WebSocket non géré:', data);
      }
    } catch (error) {
      console.error('Erreur lors du parsing du message WebSocket:', error);
      callbacks.onError?.({ type: 'parse_error', message: error });
    }
  },

  // Créer une connexion WebSocket avec gestionnaires automatiques
  createConnection: (token: string, callbacks: {
    onConnect?: () => void;
    onDisconnect?: () => void;
    onError?: (error: Event) => void;
    onWorkflowUpdate?: (data: any) => void;
    onWorkflowStatusChange?: (data: any) => void;
    onTaskUpdate?: (data: any) => void;
    onTaskProgress?: (data: any) => void;
    onVolunteerUpdate?: (data: any) => void;
    onVolunteerStatus?: (data: any) => void;
  }) => {
    const ws = websocketService.connect(token);
    
    ws.onopen = () => {
      console.log('WebSocket connecté');
      callbacks.onConnect?.();
    };
    
    ws.onclose = () => {
      console.log('WebSocket fermé');
      callbacks.onDisconnect?.();
    };
    
    ws.onerror = (error) => {
      console.error('Erreur WebSocket:', error);
      callbacks.onError?.(error);
    };
    
    ws.onmessage = (event) => {
      websocketService.handleMessage(event, callbacks);
    };
    
    return ws;
  },

  // Classe utilitaire pour gérer une connexion WebSocket
  createManager: (token: string) => {
    let ws: WebSocket | null = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelay = 1000;

    return {
      connect: (callbacks: any) => {
        ws = websocketService.createConnection(token, {
          ...callbacks,
          onDisconnect: () => {
            callbacks.onDisconnect?.();
            
            // Reconnexion automatique
            if (reconnectAttempts < maxReconnectAttempts) {
              reconnectAttempts++;
              setTimeout(() => {
                if (ws && ws.readyState === WebSocket.CLOSED) {
                  ws = websocketService.createConnection(token, callbacks);
                }
              }, reconnectDelay * reconnectAttempts);
            }
          },
          onConnect: () => {
            reconnectAttempts = 0;
            callbacks.onConnect?.();
          }
        });
        return ws;
      },
      
      disconnect: () => {
        if (ws) {
          ws.close();
          ws = null;
        }
      },
      
      send: (message: any) => {
        return ws ? websocketService.send(ws, message) : false;
      },
      
      subscribeToWorkflow: (workflowId: string) => {
        return ws ? websocketService.subscribeToWorkflow(ws, workflowId) : false;
      },
      
      subscribeToTask: (taskId: string) => {
        return ws ? websocketService.subscribeToTask(ws, taskId) : false;
      },
      
      subscribeToVolunteer: (volunteerId: string) => {
        return ws ? websocketService.subscribeToVolunteer(ws, volunteerId) : false;
      },
      
      ping: () => {
        return ws ? websocketService.ping(ws) : false;
      },
      
      isConnected: () => {
        return ws ? websocketService.isConnected(ws) : false;
      },
      
      getReadyState: () => {
        return ws ? ws.readyState : WebSocket.CLOSED;
      }
    };
  }
};

export default api;