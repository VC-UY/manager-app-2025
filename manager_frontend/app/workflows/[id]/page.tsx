'use client';

import { useEffect, useState, JSX } from 'react';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { useManagerWebSocket } from '@/hooks/useManagerWebSocket';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { workflowService, taskService } from '@/lib/api';

// Types
interface Workflow {
  id: string;
  name: string;
  description: string;
  workflow_type: string;
  status: string;
  created_at: string;
  owner: {
    username: string;
  };
  executable_path: string;
  input_path: string;
  output_path: string;
  priority: number;
  max_execution_time: number;
  retry_count: number;
  submitted_at: string;
  completed_at: string;
}

interface Task {
  id: string;
  name: string;
  status: string;
  progress: number;
}

export default function WorkflowDetailPage() {
  // Ajout d'une fonction de validation des messages WebSocket
  function validateWebSocketMessage(event: any, requiredFields: string[]) {
    for (const field of requiredFields) {
      if (!(field in event)) {
        console.error(`Champ manquant dans le message WebSocket : ${field}`);
        return false;
      }
    }
    return true;
  }

  // Affichage des événements temps réel via WebSocket
  useManagerWebSocket((event) => {
    // Validation des messages WebSocket
    const requiredFields = ['type', 'workflow_id'];
    if (!validateWebSocketMessage(event, requiredFields)) {
      toast.error('Message WebSocket invalide reçu', { position: 'top-right' });
      return;
    }

    // Toast (déjà présent)
    let msg = '';
    if (event.type) {
      msg = `[${event.type}] ` + (event.message || JSON.stringify(event));
    } else {
      msg = JSON.stringify(event);
    }
    // Toast uniquement pour les événements majeurs
    if (event.type === 'workflow_status_change' ) {
      // Afficher un toast pour tous les changements de statut
      toast.info(event.message || `[workflow_status_change] ${event.status}`, { position: 'top-right' });
      // Mettre à jour le statut du workflow (affichage DOM)
      if (event.workflow_id === id && workflow) {
        setWorkflow({ ...workflow, status: event.status });
      }
    } else if (event.type === 'task_status_change' || event.type === 'task_status_update' || event.type === 'task_update' || event.type === 'task_status') {
      if (event.status === 'COMPLETED') {
        toast.success(event.message || 'Tâche complétée', { position: 'top-right' });
      } else if (event.status === 'FAILED') {
        toast.error(event.message || 'Tâche échouée', { position: 'top-right' });
      } else if (event.status === 'STARTED' || event.status === 'RUNNING') {
        toast.info(event.message || 'Tâche démarrée', { position: 'top-right' });
      }
      if(event.status === 'TERMINATED') {
        toast.error(event.message || 'Tâche terminée', { position: 'top-right' });
      }
      if(event.status === 'SPLIT_COMPLETED') {
        toast.info(event.message || 'Decoupage terminé', { position: 'top-right' });
        // Rafraîchir la liste des tâches
        taskService.getWorkflowTasks(id as string)
          .then(setTasks)
          .catch(() => toast.error('Impossible de charger les tâches du workflow après découpage'));
      }
    }else if (event.type === 'workflow_update') {
      setWorkflow(event.workflow);
    }else {
      console.log("Event type non reconnu", event);
      toast.warning(msg, { position: 'top-right', autoClose: 10000 });
    }
      

    // Rafraîchir la liste des tâches après SPLIT_COMPLETED
    if (event.type === 'workflow_status_change' && event.status === 'SPLIT_COMPLETED' && event.workflow_id === id) {
      // Recharge aussi le workflow pour avoir le bon statut
      workflowService.getWorkflow(id as string)
        .then(setWorkflow)
        .catch(() => toast.error('Impossible de charger le workflow après découpage', { position: 'top-right' }));

      taskService.getWorkflowTasks(id as string)
        .then(setTasks)
        .catch(() => toast.error('Impossible de charger les tâches du workflow après découpage', { position: 'top-right' }));
    }

    // Progression d'une tâche
    if (event.type === 'task_progress' && event.workflow_id === id && event.task_id) {
      setTasks((prev) => prev.map(task => task.id === event.task_id ? { ...task, progress: event.progress ?? 0 } : task));
    }

    // Changement de statut d'une tâche
    if (event.type === 'task_status_change' && event.workflow_id === id && event.task_id) {
      setTasks((prev) => prev.map(task => task.id === event.task_id ? { ...task, status: event.status ?? task.status, progress: event.status === 'COMPLETED' ? 100 : task.progress } : task));
    }

    // Mise à jour silencieuse des tâches sur task_update
    if (event.type === 'task_update' && event.task && event.task.workflow_id === id) {
      setTasks((prev) => prev.map(task => task.id === event.task.id ? { ...task, ...event.task, progress: event.status === 'COMPLETED' ? 100 : task.progress } : task));
    }

    // Mise à jour silencieuse du workflow sur workflow_update
    if (event.type === 'workflow_update' && event.workflow && event.workflow.id === id) {
      setWorkflow(event.workflow);
    }
  });
  const { id } = useParams();
  const router = useRouter();

  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // --- Chargement du workflow et de ses tâches ---
  useEffect(() => {
    const fetchWorkflow = async () => {
      setLoading(true);
      try {
        // 1) Charger les détails du workflow
        const wf = await workflowService.getWorkflow(id as string);
        setWorkflow(wf);

        // 2) Charger toutes les tâches liées à ce workflow
        const workflowTasks = await taskService.getWorkflowTasks(id as string);
        setTasks(workflowTasks);

        setError(null);
      } catch (err: any) {
        console.error('Erreur lors du chargement:', err);
        setError(err.error || 'Une erreur est survenue lors du chargement');
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchWorkflow();
    }
  }, [id]);

  // --- Soumission du workflow ---
  const handleSubmit = async () => {
    if (!workflow) return;
    setSubmitting(true);
    try {
      await workflowService.submitWorkflow(workflow.id);
      // Recharger le workflow
      const updated = await workflowService.getWorkflow(workflow.id);
      setWorkflow(updated);
      // Recharger aussi les tâches (elles seront créées côté back)
      const workflowTasks = await taskService.getWorkflowTasks(workflow.id);
      setTasks(workflowTasks);
      setError(null);
    } catch (err: any) {
      console.error('Erreur lors de la soumission:', err);
      setError(err.error || 'Une erreur est survenue lors de la soumission');
    } finally {
      setSubmitting(false);
    }
  };

  // --- Formatage d’une date au format fr-FR ---
  const formatDate = (dateString?: string | null) => {
    if (!dateString) return 'Non disponible';
    return new Intl.DateTimeFormat('fr-FR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(dateString));
  };

  // Obtenir les informations de statut avec icône et couleurs
  const getStatusInfo = (status: string) => {
    switch (status) {
      case 'CREATED':
        return {
          bgColor: 'bg-gray-200',
          textColor: 'text-gray-800',
          borderColor: 'border-gray-300',
          label: 'Créé',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          )
        };
      case 'VALIDATED':
        return {
          bgColor: 'bg-blue-100',
          textColor: 'text-blue-800',
          borderColor: 'border-blue-200',
          label: 'Validé',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )
        };
      case 'SUBMITTED':
        return {
          bgColor: 'bg-yellow-100',
          textColor: 'text-yellow-800',
          borderColor: 'border-yellow-200',
          label: 'Soumis',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          )
        };
      case 'RUNNING':
        return {
          bgColor: 'bg-green-100',
          textColor: 'text-green-800',
          borderColor: 'border-green-200',
          label: 'En cours',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          )
        };
      case 'COMPLETED':
        return {
          bgColor: 'bg-green-200',
          textColor: 'text-green-800',
          borderColor: 'border-green-300',
          label: 'Terminé',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )
        };
      case 'FAILED':
        return {
          bgColor: 'bg-red-100',
          textColor: 'text-red-800',
          borderColor: 'border-red-200',
          label: 'Échoué',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )
        };
      default:
        return {
          bgColor: 'bg-gray-100',
          textColor: 'text-gray-800',
          borderColor: 'border-gray-200',
          label: status,
          icon: null
        };
    }
  };

  // Obtenir les informations sur le type de workflow
  const getWorkflowTypeInfo = (type: string) => {
    const typeMap: {[key: string]: {icon: JSX.Element, label: string, description: string}} = {
      'MATRIX_ADDITION': {
        icon: (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
        ),
        label: 'Addition de matrices',
        description: 'Effectue l\'addition de deux matrices de dimensions identiques.'
      },
      'MATRIX_MULTIPLICATION': {
        icon: (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v1m6 11h2m-6 0h-2v4m0-11v3m0 0h.01M12 12h4.01M16 20h4M4 12h4m12 0h.01M5 8h2a1 1 0 001-1V5a1 1 0 00-1-1H5a1 1 0 00-1 1v2a1 1 0 001 1zm12 0h2a1 1 0 001-1V5a1 1 0 00-1-1h-2a1 1 0 00-1 1v2a1 1 0 001 1zM5 20h2a1 1 0 001-1v-2a1 1 0 00-1-1H5a1 1 0 00-1 1v2a1 1 0 001 1z" />
          </svg>
        ),
        label: 'Multiplication de matrices',
        description: 'Effectue la multiplication de deux matrices compatibles.'
      },
      'ML_TRAINING': {
        icon: (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        ),
        label: 'Entraînement ML',
        description: 'Exécute l\'entraînement d\'un modèle de machine learning sur un jeu de données.'
      },
      'CUSTOM': {
        icon: (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        ),
        label: 'Personnalisé',
        description: 'Workflow personnalisé avec des paramètres spécifiques.'
      }
    };

    return typeMap[type] || {
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      ),
      label: type,
      description: 'Type de workflow spécifique.'
    };
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col justify-center items-center h-64 bg-white rounded-xl shadow-md">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500 mb-4"></div>
          <p className="text-blue-800 text-lg font-medium animate-pulse">Chargement des détails du workflow...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-6 rounded-lg shadow-md mb-6">
          <div className="flex items-center">
            <svg className="h-6 w-6 text-red-500 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="font-medium text-red-800">{error}</span>
          </div>
        </div>
        <div className="flex justify-center mt-6">
          <Link
            href="/workflows"
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg text-md font-medium transition-colors duration-200 flex items-center shadow-md"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
            </svg>
            Retour à la liste des workflows
          </Link>
        </div>
      </div>
    );
  }

  if (!workflow) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-8 text-center shadow-md">
          <svg className="h-16 w-16 text-blue-500 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h2 className="text-xl font-medium text-blue-800 mb-2">Workflow non trouvé</h2>
          <p className="text-blue-600 mb-6">Ce workflow n'existe pas ou a été supprimé.</p>
          <Link
            href="/workflows"
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg text-md font-medium transition-colors duration-200 inline-flex items-center shadow-md"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
            </svg>
            Retour à la liste des workflows
          </Link>
        </div>
      </div>
    );
  }

  const statusInfo = getStatusInfo(workflow.status);
  const typeInfo = getWorkflowTypeInfo(workflow.workflow_type);

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      {/* Bannière en haut avec statut et actions */}
      <div className="bg-gradient-to-r from-blue-700 to-indigo-800 rounded-xl shadow-lg mb-8 overflow-hidden relative">
        <div className="absolute inset-0 bg-grid-white/10 opacity-10"></div>
        <div className="relative z-10 px-8 py-6 text-white">
          <div className="flex flex-col md:flex-row md:justify-between md:items-center space-y-4 md:space-y-0">
            <div>
              <div className="flex items-center">
                <Link
                  href="/workflows"
                  className="mr-3 bg-white/10 hover:bg-white/20 p-2 rounded-full transition-colors duration-200"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                  </svg>
                </Link>
                <h1 className="text-2xl md:text-3xl font-bold">{workflow.name}</h1>
              </div>
              <div className="flex flex-wrap items-center mt-3 gap-2">
                <span className={`flex items-center px-3 py-1 text-sm font-medium rounded-full ${statusInfo.bgColor} ${statusInfo.textColor} border ${statusInfo.borderColor}`}>
                  {statusInfo.icon}
                  {statusInfo.label}
                </span>
                <span className="bg-white/10 text-white px-3 py-1 text-sm rounded-full backdrop-blur-sm inline-flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  Créé le {formatDate(workflow.created_at)}
                </span>
                <span className="bg-white/10 text-white px-3 py-1 text-sm rounded-full backdrop-blur-sm inline-flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                  {workflow.owner?.username || 'Non assigné'}
                </span>
              </div>
            </div>
            <div className="flex gap-3">
              <Link
                href={`/workflows/${workflow.id}/edit`}
                className="bg-white/10 hover:bg-white/20 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center backdrop-blur-sm border border-white/10"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                Modifier
              </Link>
              {workflow.status === 'CREATED' && (
                <button
                  onClick={handleSubmit}
                  disabled={submitting}
                  className={`bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center shadow-md ${
                    submitting ? 'opacity-70 cursor-not-allowed' : ''
                  }`}
                >
                  {submitting ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Soumission...
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Soumettre
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Barre de progression du workflow */}
          {workflow.status !== 'CREATED' && workflow.status !== 'FAILED' && (
            <div className="mt-6">
              <div className="flex justify-between mb-1 text-sm text-blue-100">
                <span>Progression</span>
                <span>
                  {workflow.status === 'COMPLETED' 
                    ? '100%' 
                    : workflow.status === 'RUNNING' 
                      ? '75%' 
                      : workflow.status === 'SUBMITTED' 
                        ? '25%' 
                        : '0%'}
                </span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2.5 backdrop-blur-sm">
                <div 
                  className="bg-blue-500 h-2.5 rounded-full"
                  style={{ 
                    width: workflow.status === 'COMPLETED' 
                      ? '100%' 
                      : workflow.status === 'RUNNING' 
                        ? '75%' 
                        : workflow.status === 'SUBMITTED' 
                          ? '25%' 
                          : '0%' 
                  }}
                ></div>
              </div>
            </div>
          )}
        </div>
        
        {/* Vagues décoratives */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320">
            <path fill="rgba(255,255,255,0.1)" fillOpacity="1" d="M0,288L48,272C96,256,192,224,288,197.3C384,171,480,149,576,165.3C672,181,768,235,864,250.7C960,267,1056,245,1152,224C1248,203,1344,181,1392,170.7L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
          </svg>
        </div>
      </div>

      {/* Carte d'aperçu */}
      <div className="mb-6 bg-white rounded-xl shadow-md overflow-hidden border border-gray-100">
        <ToastContainer />
        <div className="p-6">
          <div className="flex items-center mb-4">
            <div className="p-3 rounded-full bg-blue-100 mr-4">
              {typeInfo.icon}
            </div>
            <div>
              <h2 className="text-xl font-bold text-blue-900">{typeInfo.label}</h2>
              <p className="text-blue-800">{typeInfo.description}</p>
            </div>
          </div>
          
          <div className="mt-2">
            <p className="text-blue-900 mb-4">
              {workflow.description || 'Aucune description fournie pour ce workflow.'}
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
                <h3 className="text-md font-semibold text-blue-800 mb-2 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Priorité
                </h3>
                <div className="text-blue-900 text-xl font-bold">{workflow.priority}</div>
                <p className="text-blue-700 text-sm mt-1">
                  {workflow.priority > 8 
                    ? 'Haute priorité' 
                    : workflow.priority > 5 
                      ? 'Priorité moyenne' 
                      : 'Priorité basse'}
                </p>
              </div>
              
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
                <h3 className="text-md font-semibold text-blue-800 mb-2 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Temps d'exécution
                </h3>
                <div className="text-blue-900 text-xl font-bold">{workflow.max_execution_time} s</div>
                <p className="text-blue-700 text-sm mt-1">
                  Temps d'exécution maximum autorisé
                </p>
              </div>
              
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
                <h3 className="text-md font-semibold text-blue-800 mb-2 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Tentatives
                </h3>
                <div className="text-blue-900 text-xl font-bold">{workflow.retry_count}</div>
                <p className="text-blue-700 text-sm mt-1">
                  Nombre de tentatives en cas d'échec
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
        
      {/* Détails du workflow en tabs */}
      <div className="bg-white rounded-xl shadow-md overflow-hidden mb-6 border border-gray-100">
        <div className="border-b border-gray-200">
          <ul className="flex overflow-x-auto">
            <li className="text-blue-600 border-b-2 border-blue-500 px-4 py-4 font-medium">
              Détails de configuration
            </li>
          </ul>
        </div>
        
        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
                Configuration technique
              </h3>
              
              <div className="space-y-4">
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
                  <h4 className="text-md font-medium text-blue-800 mb-2">Chemin de l'exécutable</h4>
                  <div className="bg-white text-blue-900 p-3 rounded-md border border-blue-200 font-mono text-sm overflow-x-auto">
                    {workflow.executable_path || 'Non spécifié'}
                  </div>
                </div>
                
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
                  <h4 className="text-md font-medium text-blue-800 mb-2">Chemin des données d'entrée</h4>
                  <div className="bg-white text-blue-900 p-3 rounded-md border border-blue-200 font-mono text-sm overflow-x-auto">
                    {workflow.input_path || 'Non spécifié'}
                  </div>
                </div>
                
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
                  <h4 className="text-md font-medium text-blue-800 mb-2">Chemin des résultats</h4>
                  <div className="bg-white text-blue-900 p-3 rounded-md border border-blue-200 font-mono text-sm overflow-x-auto">
                    {workflow.output_path || 'Non spécifié'}
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                Chronologie d'exécution
              </h3>
              
              <div className="relative pl-8 border-l-2 border-blue-200 space-y-6">
                <div className="relative">
                  <div className="absolute -left-10 top-1 rounded-full w-6 h-6 bg-blue-500 text-white flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                  </div>
                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <h4 className="text-md font-medium text-blue-800 mb-1">Date de création</h4>
                    <p className="text-blue-900">{formatDate(workflow.created_at)}</p>
                  </div>
                </div>
                
                <div className="relative">
                  <div className={`absolute -left-10 top-1 rounded-full w-6 h-6 ${workflow.submitted_at ? 'bg-blue-500' : 'bg-gray-300'} text-white flex items-center justify-center`}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </div>
                  <div className={`${workflow.submitted_at ? 'bg-blue-50 border-blue-100' : 'bg-gray-50 border-gray-200'} p-4 rounded-lg border`}>
                    <h4 className={`text-md font-medium ${workflow.submitted_at ? 'text-blue-800' : 'text-gray-600'} mb-1`}>Date de soumission</h4>
                    <p className={workflow.submitted_at ? 'text-blue-900' : 'text-gray-600'}>
                      {workflow.submitted_at ? formatDate(workflow.submitted_at) : 'Non soumis'}
                    </p>
                  </div>
                </div>
                
                <div className="relative">
                  <div className={`absolute -left-10 top-1 rounded-full w-6 h-6 ${workflow.completed_at ? 'bg-blue-500' : 'bg-gray-300'} text-white flex items-center justify-center`}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div className={`${workflow.completed_at ? 'bg-blue-50 border-blue-100' : 'bg-gray-50 border-gray-200'} p-4 rounded-lg border`}>
                    <h4 className={`text-md font-medium ${workflow.completed_at ? 'text-blue-800' : 'text-gray-600'} mb-1`}>Date de complétion</h4>
                    <p className={workflow.completed_at ? 'text-blue-900' : 'text-gray-600'}>
                      {workflow.completed_at ? formatDate(workflow.completed_at) : 'Non terminé'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Section des tâches */}
      <div className="bg-white rounded-xl shadow-md overflow-hidden mb-6 border border-gray-100">
        <div className="border-b border-gray-200 px-6 py-4 flex justify-between items-center">
          <h2 className="text-xl font-bold text-blue-900">Tâches</h2>
          {tasks.length > 0 && (
            <span className="bg-blue-100 text-blue-800 text-sm font-medium px-3 py-1 rounded-full">
              {tasks.filter(task => task.status === 'COMPLETED').length} / {tasks.length} terminées
            </span>
          )}
        </div>
        
        {tasks.length === 0 ? (
          <div className="text-center py-12 px-6">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-blue-300 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <h3 className="text-lg font-medium text-blue-900 mb-2">
              {workflow.status === 'CREATED' 
                ? 'Les tâches seront générées après la soumission du workflow.' 
                : 'Aucune tâche trouvée pour ce workflow.'}
            </h3>
            <p className="text-blue-700 max-w-md mx-auto">
              {workflow.status === 'CREATED' 
                ? 'Une fois le workflow soumis, les tâches associées apparaîtront ici.' 
                : 'Vérifiez si le workflow a été configuré correctement.'}
            </p>
          </div>
        ) : (
          <div className="p-6">
            <div className="grid grid-cols-1 gap-4">
              {tasks.map((task) => {
                const taskStatusInfo = getStatusInfo(task.status);
                
                return (
                  <div key={task.id} className="border border-blue-100 rounded-lg overflow-hidden bg-blue-50 hover:shadow-md transition-shadow duration-200">
                    <div className="p-4">
                      <div className="flex justify-between items-start">
                        <div className="flex items-center">
                          <div className={`rounded-full w-10 h-10 flex items-center justify-center ${
                            task.status === 'COMPLETED'
                              ? 'bg-green-100 text-green-700'
                              : task.status === 'RUNNING'
                                ? 'bg-blue-100 text-blue-700'
                                : 'bg-yellow-100 text-yellow-700'
                          }`}>
                            {task.status === 'COMPLETED' ? (
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                            ) : task.status === 'RUNNING' ? (
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                              </svg>
                            ) : (
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                            )}
                          </div>
                          <div className="ml-4">
                            <h3 className="text-lg font-medium text-blue-900">{task.name}</h3>
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${taskStatusInfo.bgColor} ${taskStatusInfo.textColor} mt-1`}>
                              {taskStatusInfo.icon}
                              {taskStatusInfo.label}
                            </span>
                          </div>
                        </div>
                        <Link
                          href={`/workflows/${id}/tasks/${task.id}`}
                          className="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center bg-white px-3 py-1 rounded-md border border-blue-200 shadow-sm"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                          </svg>
                          Détails
                        </Link>
                      </div>
                      
                      <div className="mt-4">
                        <div className="flex justify-between mb-1 text-sm font-medium text-blue-800">
                          <span>Progression</span>
                          <span>{task.progress}%</span>
                        </div>
                        <div className="w-full bg-white rounded-full h-2.5 mb-4 border border-blue-200">
                          <div 
                            className={`h-2.5 rounded-full ${
                              task.status === 'COMPLETED' 
                                ? 'bg-green-500' 
                                : 'bg-blue-600'
                            }`}
                            style={{ width: `${task.progress}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Lien de retour */}
      <div className="flex justify-between items-center mt-8">
        <Link
          href="/workflows"
          className="text-blue-600 hover:text-blue-800 flex items-center font-medium"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
          </svg>
          Retour à la liste des workflows
        </Link>
        
        {workflow.status === 'COMPLETED' && (
          <Link
            href={`/workflows/${workflow.id}/results`}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Voir les résultats
          </Link>
        )}
      </div>
    </div>
  );
}