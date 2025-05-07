import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Token ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Fonctions pour les workflows
const fetchWorkflows = async (filters = {}) => {
  try {
    // Construire les paramètres de requête à partir des filtres
    let queryParams = new URLSearchParams();
    
    if (filters.status) {
      queryParams.append('status', filters.status);
    }
    
    if (filters.type) {
      queryParams.append('type', filters.type);
    }
    
    if (filters.tag) {
      queryParams.append('tag', filters.tag);
    }
    
    const url = `/workflows/${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    const response = await api.get(url);
    return response.data;
  } catch (error) {
    console.error('Error fetching workflows:', error);
    throw error;
  }
};

const fetchWorkflowDetails = async (id) => {
  try {
    const response = await api.get(`/workflows/${id}/`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching workflow ${id}:`, error);
    throw error;
  }
};

const createMatrixWorkflow = async (workflowData) => {
  try {
    // Création d'un workflow matriciel
    const requestData = {
      name: workflowData.name,
      description: workflowData.description,
      workflow_type: workflowData.type || "MATRIX_ADDITION", // MATRIX_ADDITION ou MATRIX_MULTIPLICATION
      tags: workflowData.tags || [],
      input_data: {
        matrix_a: {
          format: workflowData.matrixAFormat || "numpy",
          dimensions: workflowData.matrixADimensions || [100, 100],
          storage_type: "embedded",
          data: workflowData.matrixAData || generateRandomMatrix(workflowData.matrixADimensions),
        },
        matrix_b: {
          format: workflowData.matrixBFormat || "numpy",
          dimensions: workflowData.matrixBDimensions || [100, 100],
          storage_type: "embedded",
          data: workflowData.matrixBData || generateRandomMatrix(workflowData.matrixBDimensions),
        },
        algorithm: workflowData.algorithm || "standard",
        precision: workflowData.precision || "double",
        block_size: workflowData.blockSize || 100
      },
      min_volunteers: workflowData.minVolunteers || 1,
      max_volunteers: workflowData.maxVolunteers || 10,
      volunteer_preferences: workflowData.volunteerPreferences || []
    };
    
    const response = await api.post('/workflows/', requestData);
    return response.data;
  } catch (error) {
    console.error('Error creating matrix workflow:', error);
    throw error;
  }
};

const submitWorkflow = async (id) => {
  try {
    const response = await api.post(`/workflows/${id}/submit/`);
    return response.data;
  } catch (error) {
    console.error(`Error submitting workflow ${id}:`, error);
    throw error;
  }
};

const containerizeWorkflow = async (id) => {
  try {
    const response = await api.post(`/workflows/${id}/containerize/`);
    return response.data;
  } catch (error) {
    console.error(`Error containerizing workflow ${id}:`, error);
    throw error;
  }
};

const pushWorkflowImages = async (id) => {
  try {
    const response = await api.post(`/workflows/${id}/push-images/`);
    return response.data;
  } catch (error) {
    console.error(`Error pushing Docker images for workflow ${id}:`, error);
    throw error;
  }
};

const fetchWorkflowTasks = async (id, status = null) => {
  try {
    const params = status ? `?status=${status}` : '';
    const response = await api.get(`/workflows/${id}/tasks/${params}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching tasks for workflow ${id}:`, error);
    throw error;
  }
};

const fetchWorkflowResultInfo = async (id) => {
  try {
    const response = await api.get(`/workflows/${id}/result-info/`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching result info for workflow ${id}:`, error);
    throw error;
  }
};

const aggregateWorkflowResults = async (id) => {
  try {
    const response = await api.post(`/workflows/${id}/aggregate-results/`);
    return response.data;
  } catch (error) {
    console.error(`Error aggregating results for workflow ${id}:`, error);
    throw error;
  }
};

const pauseWorkflow = async (id) => {
  try {
    const response = await api.post(`/workflows/${id}/pause/`);
    return response.data;
  } catch (error) {
    console.error(`Error pausing workflow ${id}:`, error);
    throw error;
  }
};

const resumeWorkflow = async (id) => {
  try {
    const response = await api.post(`/workflows/${id}/resume/`);
    return response.data;
  } catch (error) {
    console.error(`Error resuming workflow ${id}:`, error);
    throw error;
  }
};

const cancelWorkflow = async (id) => {
  try {
    const response = await api.post(`/workflows/${id}/cancel/`);
    return response.data;
  } catch (error) {
    console.error(`Error canceling workflow ${id}:`, error);
    throw error;
  }
};

const fetchDashboardData = async () => {
  try {
    // Récupérer les données des workflows
    const workflowsResponse = await api.get('/workflows/');
    const workflowsData = workflowsResponse.data;
    
    // S'assurer que nous avons un tableau à traiter (gestion de différents formats de réponse API)
    const workflows = Array.isArray(workflowsData) ? workflowsData : 
                     (workflowsData && typeof workflowsData === 'object' && Array.isArray(workflowsData.results) ? 
                      workflowsData.results : []);
    
    return {
      totalWorkflows: workflows.length,
      activeWorkflows: workflows.filter(w => w && w.status && ['SUBMITTED', 'SPLITTING', 'ASSIGNING', 'PENDING', 'RUNNING'].includes(w.status)).length,
      completedWorkflows: workflows.filter(w => w && w.status === 'COMPLETED').length,
      failedWorkflows: workflows.filter(w => w && w.status && ['FAILED', 'PARTIAL_FAILURE'].includes(w.status)).length,
      recentWorkflows: workflows.slice(0, 5),
      // Statistiques sur les types de matrices
      matrixAdditions: workflows.filter(w => w && w.workflow_type === 'MATRIX_ADDITION').length,
      matrixMultiplications: workflows.filter(w => w && w.workflow_type === 'MATRIX_MULTIPLICATION').length
    };
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    // Retourner des données par défaut en cas d'erreur
    return {
      totalWorkflows: 0,
      activeWorkflows: 0,
      completedWorkflows: 0,
      failedWorkflows: 0,
      recentWorkflows: [],
      matrixAdditions: 0,
      matrixMultiplications: 0
    };
  }
};

// Fonctions pour les tâches
const fetchTask = async (taskId) => {
  try {
    const response = await api.get(`/tasks/${taskId}/`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching task ${taskId}:`, error);
    throw error;
  }
};

const assignTask = async (taskId, volunteerId = null) => {
  try {
    const response = await api.post(`/tasks/${taskId}/assign/`, {
      volunteer_id: volunteerId
    });
    return response.data;
  } catch (error) {
    console.error(`Error assigning task ${taskId}:`, error);
    throw error;
  }
};

const abortTask = async (taskId, errorDetails = {}) => {
  try {
    const response = await api.post(`/tasks/${taskId}/abort/`, {
      error_details: errorDetails
    });
    return response.data;
  } catch (error) {
    console.error(`Error aborting task ${taskId}:`, error);
    throw error;
  }
};

const reassignTask = async (taskId, volunteerId = null) => {
  try {
    const response = await api.post(`/tasks/${taskId}/reassign/`, {
      volunteer_id: volunteerId
    });
    return response.data;
  } catch (error) {
    console.error(`Error reassigning task ${taskId}:`, error);
    throw error;
  }
};

const aggregateTaskResults = async (taskId) => {
  try {
    const response = await api.post(`/tasks/${taskId}/aggregate-results/`);
    return response.data;
  } catch (error) {
    console.error(`Error aggregating results for task ${taskId}:`, error);
    throw error;
  }
};

const fetchTaskMatrixInfo = async (taskId) => {
  try {
    const response = await api.get(`/tasks/${taskId}/matrix-info/`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching matrix info for task ${taskId}:`, error);
    throw error;
  }
};

const fetchSubtasks = async (taskId) => {
  try {
    const response = await api.get(`/tasks/${taskId}/subtasks/`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching subtasks for task ${taskId}:`, error);
    throw error;
  }
};

const retryFailedSubtasks = async (taskId) => {
  try {
    const response = await api.post(`/tasks/${taskId}/retry-failed-subtasks/`);
    return response.data;
  } catch (error) {
    console.error(`Error retrying failed subtasks for task ${taskId}:`, error);
    throw error;
  }
};

// Fonctions utilitaires
function generateRandomMatrix(dimensions) {
  if (!dimensions || !Array.isArray(dimensions) || dimensions.length !== 2) {
    dimensions = [10, 10]; // Dimensions par défaut
  }
  
  const [rows, cols] = dimensions;
  const matrix = [];
  
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      // Générer des nombres aléatoires entre -10 et 10
      row.push(Math.round((Math.random() * 20 - 10) * 100) / 100);
    }
    matrix.push(row);
  }
  
  return matrix;
}

export {
  fetchWorkflows,
  fetchWorkflowDetails,
  createMatrixWorkflow,
  submitWorkflow,
  containerizeWorkflow,
  pushWorkflowImages,
  fetchWorkflowTasks,
  fetchWorkflowResultInfo,
  aggregateWorkflowResults,
  pauseWorkflow,
  resumeWorkflow,
  cancelWorkflow,
  fetchDashboardData,
  // Fonctions pour les tâches
  fetchTask,
  assignTask,
  abortTask,
  reassignTask,
  aggregateTaskResults,
  fetchTaskMatrixInfo,
  fetchSubtasks,
  retryFailedSubtasks,
  // Fonction utilitaire
  generateRandomMatrix
};

export default api;