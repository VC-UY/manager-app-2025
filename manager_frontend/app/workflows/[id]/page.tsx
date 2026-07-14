'use client';

import { useCallback, useEffect, useState, JSX } from 'react';
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

const RESUBMITTABLE = new Set(['FAILED', 'PARTIAL_FAILURE', 'ERROR', 'PENDING']);

export default function WorkflowDetailPage() {
  const { id } = useParams();
  const router = useRouter();

  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshWorkflow = useCallback(async () => {
    if (!id) return;
    try {
      const [wf, workflowTasks] = await Promise.all([
        workflowService.getWorkflow(id as string),
        taskService.getWorkflowTasks(id as string),
      ]);
      setWorkflow(wf);
      setTasks(Array.isArray(workflowTasks) ? workflowTasks : workflowTasks?.results || []);
    } catch {
      // silencieux en temps réel
    }
  }, [id]);

  useManagerWebSocket({
    workflowId: id as string,
    showToasts: true,
    enabled: Boolean(id),
    handlers: {
      onWorkflowStatus: (event) => {
        if (event.status) {
          setWorkflow((prev) => (prev ? { ...prev, status: event.status as string } : prev));
        }
        if (
          event.status === 'SPLIT_COMPLETED' ||
          event.status === 'PENDING' ||
          event.status === 'RUNNING' ||
          event.status === 'COMPLETED' ||
          event.status === 'FAILED'
        ) {
          refreshWorkflow();
        }
      },
      onWorkflowUpdate: (event) => {
        const wf = event.workflow;
        if (wf && typeof wf === 'object' && wf.id) {
          setWorkflow((prev) => (prev ? { ...prev, ...wf, status: (wf.status as string) || prev.status } : prev));
        }
      },
      onTaskStatus: (event) => {
        const taskId = event.task_id || event.task?.id;
        const status = event.status || event.task?.status;
        if (taskId && status) {
          const terminal = new Set(['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']);
          setTasks((prev) =>
            prev.map((task) => {
              if (task.id !== taskId) return task;
              // Ne jamais rétrograder une tâche terminée (WS stale à 2%)
              if (terminal.has(String(task.status || '').toUpperCase())
                  && !terminal.has(String(status).toUpperCase())) {
                return task;
              }
              return {
                ...task,
                status: status as string,
                progress: status === 'COMPLETED' ? 100 : task.progress,
              };
            }),
          );
        }
        if (status === 'COMPLETED' || status === 'FAILED') {
          refreshWorkflow();
        }
      },
      onTaskProgress: (event) => {
        if (event.task_id != null && event.progress != null) {
          const incoming = Number(event.progress) || 0;
          setTasks((prev) =>
            prev.map((task) => {
              if (task.id !== event.task_id) return task;
              const curStatus = String(task.status || '').toUpperCase();
              if (['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'].includes(curStatus)) {
                return task;
              }
              // Progression monotone
              const cur = Number(task.progress) || 0;
              if (incoming + 0.5 < cur && cur >= 50) return task;
              return { ...task, progress: Math.max(cur, incoming) };
            }),
          );
        }
      },
    },
  });

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
        setTasks(Array.isArray(workflowTasks) ? workflowTasks : workflowTasks?.results || []);

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

  // --- Soumission / resoumission du workflow ---
  const handleSubmit = async (resubmit = false) => {
    if (!workflow) return;
    setSubmitting(true);
    try {
      const result = resubmit
        ? await workflowService.resubmitWorkflow(workflow.id)
        : await workflowService.submitWorkflow(workflow.id);
      const msg =
        (result && (result.message || result.detail)) ||
        (resubmit
          ? 'Workflow réinitialisé et resoumis. Les tâches iront en file d\'attente.'
          : 'Workflow soumis. Les tâches seront mises en file d\'attente puis assignées progressivement.');
      toast.success(typeof msg === 'string' ? msg : 'Workflow soumis.', { position: 'top-right' });
      await refreshWorkflow();
      setError(null);
    } catch (err: any) {
      console.error('Erreur lors de la soumission:', err);
      const msg =
        err.error ||
        err.message ||
        err.response?.message ||
        err.detail ||
        'Une erreur est survenue lors de la soumission';
      const text = typeof msg === 'string' ? msg : 'Une erreur est survenue lors de la soumission';
      setError(text);
      toast.error(text, { position: 'top-right' });
    } finally {
      setSubmitting(false);
    }
  };

  // --- Formatage d'une date au format fr-FR ---
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
  const getStatusInfo = (status: string, kind: 'workflow' | 'task' = 'workflow') => {
    const statusMap: Record<string, { color: string; bg: string; label: string }> = {
      CREATED:
        kind === 'task'
          ? { color: '#FBBF24', bg: 'rgba(251, 191, 36, 0.1)', label: "En file d'attente" }
          : { color: '#94A3B8', bg: 'rgba(148, 163, 184, 0.1)', label: 'Créé' },
      VALIDATED: { color: '#60A5FA', bg: 'rgba(96, 165, 250, 0.1)', label: 'Validé' },
      SUBMITTED: { color: '#60A5FA', bg: 'rgba(96, 165, 250, 0.1)', label: 'Soumis' },
      SPLITTING: { color: '#38BDF8', bg: 'rgba(56, 189, 248, 0.12)', label: 'Découpage…' },
      ASSIGNING: { color: '#38BDF8', bg: 'rgba(56, 189, 248, 0.12)', label: 'Attribution…' },
      PENDING: { color: '#F59E0B', bg: 'rgba(245, 158, 11, 0.12)', label: 'En attente de volontaires' },
      ASSIGNED: { color: '#38BDF8', bg: 'rgba(56, 189, 248, 0.12)', label: 'Assignée' },
      RUNNING: { color: '#3B82F6', bg: 'rgba(59, 130, 246, 0.15)', label: 'En cours' },
      AGGREGATING: { color: '#8B5CF6', bg: 'rgba(139, 92, 246, 0.12)', label: 'Agrégation…' },
      COMPLETED: { color: '#10B981', bg: 'rgba(16, 185, 129, 0.1)', label: 'Terminé' },
      FAILED: { color: '#EF4444', bg: 'rgba(239, 68, 68, 0.1)', label: 'Échoué' },
      PARTIAL_FAILURE: { color: '#F97316', bg: 'rgba(249, 115, 22, 0.12)', label: 'Échec partiel' },
      ERROR: { color: '#EF4444', bg: 'rgba(239, 68, 68, 0.1)', label: 'Erreur' },
      PAUSED: { color: '#94A3B8', bg: 'rgba(148, 163, 184, 0.1)', label: 'En pause' },
    };
    return statusMap[status] || statusMap.CREATED;
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
      <div className="min-h-screen" style={{
        background: 'linear-gradient(180deg, #0A1628 0%, #1A2942 50%, #0F1C2E 100%)'
      }}>
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col justify-center items-center h-64 rounded-xl" style={{
            background: 'rgba(30, 41, 59, 0.4)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(71, 85, 105, 0.3)'
          }}>
            <div className="animate-spin rounded-full h-16 w-16 mb-4" style={{
              border: '4px solid rgba(59, 130, 246, 0.2)',
              borderTopColor: '#3B82F6'
            }}></div>
            <p style={{ color: '#CBD5E1', fontSize: '16px', fontWeight: 500 }}>Chargement des détails du workflow...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen" style={{
        background: 'linear-gradient(180deg, #0A1628 0%, #1A2942 50%, #0F1C2E 100%)'
      }}>
        <div className="container mx-auto px-4 py-8">
          <div className="p-6 rounded-xl mb-6" style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            backdropFilter: 'blur(12px)'
          }}>
            <div className="flex items-center gap-3">
              <svg className="h-6 w-6" style={{ color: '#EF4444' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span style={{ color: '#FCA5A5', fontWeight: 500 }}>{error}</span>
            </div>
          </div>
          <div className="flex justify-center mt-6">
            <Link
              href="/workflows"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-300"
              style={{
                background: 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
                border: '1px solid rgba(59, 130, 246, 0.4)',
                color: '#FFFFFF'
              }}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
              </svg>
              Retour à la liste des workflows
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (!workflow) {
    return (
      <div className="min-h-screen" style={{
        background: 'linear-gradient(180deg, #0A1628 0%, #1A2942 50%, #0F1C2E 100%)'
      }}>
        <div className="container mx-auto px-4 py-8">
          <div className="p-8 text-center rounded-xl" style={{
            background: 'rgba(30, 41, 59, 0.4)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(71, 85, 105, 0.3)'
          }}>
            <svg className="h-16 w-16 mx-auto mb-4" style={{ color: '#60A5FA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h2 className="text-xl font-medium mb-2" style={{ color: '#FFFFFF' }}>Workflow non trouvé</h2>
            <p className="mb-6" style={{ color: '#94A3B8' }}>Ce workflow n'existe pas ou a été supprimé.</p>
            <Link
              href="/workflows"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-300"
              style={{
                background: 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
                border: '1px solid rgba(59, 130, 246, 0.4)',
                color: '#FFFFFF'
              }}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
              </svg>
              Retour à la liste des workflows
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const statusInfo = getStatusInfo(workflow.status);
  const typeInfo = getWorkflowTypeInfo(workflow.workflow_type);

  return (
    <div className="min-h-screen" style={{
      background: 'linear-gradient(180deg, #0A1628 0%, #1A2942 50%, #0F1C2E 100%)'
    }}>
      <ToastContainer />
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Bannière en haut avec statut et actions */}
        <div className="mb-8 p-8 rounded-2xl relative overflow-hidden" style={{
          background: 'rgba(30, 41, 59, 0.4)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(71, 85, 105, 0.3)',
          boxShadow: '0 4px 24px rgba(0, 0, 0, 0.3)'
        }}>
          <div className="relative z-10">
            <div className="flex flex-col md:flex-row md:justify-between md:items-center space-y-4 md:space-y-0">
              <div>
                <div className="flex items-center mb-3">
                  <Link
                    href="/workflows"
                    className="mr-3 p-2 rounded-full transition-colors duration-300" style={{
                      background: 'rgba(59, 130, 246, 0.2)',
                      border: '1px solid rgba(59, 130, 246, 0.3)'
                    }}
                  >
                    <svg className="w-5 h-5" style={{ color: '#60A5FA' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                    </svg>
                  </Link>
                  <h1 className="text-2xl md:text-3xl font-bold" style={{ color: '#FFFFFF' }}>{workflow.name}</h1>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center px-3 py-1 text-sm font-medium rounded-lg" style={{
                    background: statusInfo.bg,
                    color: statusInfo.color,
                    border: `1px solid ${statusInfo.color}40`
                  }}>
                    {statusInfo.label}
                  </span>
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 text-sm rounded-lg" style={{
                    background: 'rgba(59, 130, 246, 0.1)',
                    color: '#60A5FA',
                    border: '1px solid rgba(59, 130, 246, 0.3)'
                  }}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Créé le {formatDate(workflow.created_at)}
                  </span>
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 text-sm rounded-lg" style={{
                    background: 'rgba(59, 130, 246, 0.1)',
                    color: '#60A5FA',
                    border: '1px solid rgba(59, 130, 246, 0.3)'
                  }}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                    {workflow.owner?.username || 'Non assigné'}
                  </span>
                </div>
              </div>
              <div className="flex gap-3 flex-wrap">
                {(workflow.status === 'CREATED' || RESUBMITTABLE.has(workflow.status)) && (
                  <Link
                    href={`/workflows/${workflow.id}/edit`}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300" style={{
                      background: 'rgba(59, 130, 246, 0.2)',
                      border: '1px solid rgba(59, 130, 246, 0.3)',
                      color: '#60A5FA'
                    }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                    Modifier
                  </Link>
                )}
                {workflow.status === 'CREATED' && (
                  <button
                    onClick={() => handleSubmit(false)}
                    disabled={submitting}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300"
                    style={{
                      background: submitting ? 'rgba(59, 130, 246, 0.3)' : 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
                      border: '1px solid rgba(59, 130, 246, 0.4)',
                      color: '#FFFFFF',
                      cursor: submitting ? 'not-allowed' : 'pointer',
                      opacity: submitting ? 0.7 : 1
                    }}
                  >
                    {submitting ? (
                      <>
                        <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Soumission...
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        Soumettre
                      </>
                    )}
                  </button>
                )}
                {RESUBMITTABLE.has(workflow.status) && (
                  <button
                    onClick={() => handleSubmit(true)}
                    disabled={submitting}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300"
                    style={{
                      background: submitting
                        ? 'rgba(245, 158, 11, 0.3)'
                        : 'linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%)',
                      border: '1px solid rgba(245, 158, 11, 0.5)',
                      color: '#0F172A',
                      cursor: submitting ? 'not-allowed' : 'pointer',
                      opacity: submitting ? 0.7 : 1,
                    }}
                  >
                    {submitting ? (
                      <>
                        <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Resoumission...
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        Resoumettre
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>

            {/* Barre de progression du workflow */}
            {workflow.status !== 'CREATED' && workflow.status !== 'FAILED' && (
              <div className="mt-6">
                <div className="flex justify-between mb-2 text-sm" style={{ color: '#94A3B8' }}>
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
                <div className="w-full h-2 rounded-full overflow-hidden" style={{
                  background: 'rgba(15, 23, 42, 0.5)'
                }}>
                  <div 
                    className="h-full rounded-full transition-all duration-500"
                    style={{ 
                      width: workflow.status === 'COMPLETED' 
                        ? '100%' 
                        : workflow.status === 'RUNNING' 
                          ? '75%' 
                          : workflow.status === 'SUBMITTED' 
                            ? '25%' 
                            : '0%',
                      background: 'linear-gradient(90deg, #3B82F6 0%, #60A5FA 100%)'
                    }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Carte d'aperçu */}
        <div className="mb-6 rounded-xl overflow-hidden" style={{
          background: 'rgba(30, 41, 59, 0.4)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(71, 85, 105, 0.3)'
        }}>
          <div className="p-6">
            <div className="flex items-center mb-4">
              <div className="p-3 rounded-full mr-4" style={{
                background: 'rgba(59, 130, 246, 0.2)',
                border: '1px solid rgba(59, 130, 246, 0.3)'
              }}>
                {typeInfo.icon}
              </div>
              <div>
                <h2 className="text-xl font-bold" style={{ color: '#FFFFFF' }}>{typeInfo.label}</h2>
                <p style={{ color: '#94A3B8' }}>{typeInfo.description}</p>
              </div>
            </div>
            
            <div className="mt-2">
              <p className="mb-4" style={{ color: '#CBD5E1' }}>
                {workflow.description || 'Aucune description fournie pour ce workflow.'}
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
                <div className="rounded-lg p-4" style={{
                  background: 'rgba(59, 130, 246, 0.1)',
                  border: '1px solid rgba(59, 130, 246, 0.2)'
                }}>
                  <h3 className="text-md font-semibold mb-2 flex items-center" style={{ color: '#CBD5E1' }}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" style={{ color: '#60A5FA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Priorité
                  </h3>
                  <div className="text-xl font-bold" style={{ color: '#FFFFFF' }}>{workflow.priority}</div>
                  <p className="text-sm mt-1" style={{ color: '#94A3B8' }}>
                    {workflow.priority > 8 
                      ? 'Haute priorité' 
                      : workflow.priority > 5 
                        ? 'Priorité moyenne' 
                        : 'Priorité basse'}
                  </p>
                </div>
                
                <div className="rounded-lg p-4" style={{
                  background: 'rgba(59, 130, 246, 0.1)',
                  border: '1px solid rgba(59, 130, 246, 0.2)'
                }}>
                  <h3 className="text-md font-semibold mb-2 flex items-center" style={{ color: '#CBD5E1' }}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" style={{ color: '#60A5FA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Temps d'exécution
                  </h3>
                  <div className="text-xl font-bold" style={{ color: '#FFFFFF' }}>{workflow.max_execution_time} s</div>
                  <p className="text-sm mt-1" style={{ color: '#94A3B8' }}>
                    Temps d'exécution maximum autorisé
                  </p>
                </div>
                
                <div className="rounded-lg p-4" style={{
                  background: 'rgba(59, 130, 246, 0.1)',
                  border: '1px solid rgba(59, 130, 246, 0.2)'
                }}>
                  <h3 className="text-md font-semibold mb-2 flex items-center" style={{ color: '#CBD5E1' }}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" style={{ color: '#60A5FA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Tentatives
                  </h3>
                  <div className="text-xl font-bold" style={{ color: '#FFFFFF' }}>{workflow.retry_count}</div>
                  <p className="text-sm mt-1" style={{ color: '#94A3B8' }}>
                    Nombre de tentatives en cas d'échec
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
          
        {/* Détails du workflow */}
        <div className="rounded-xl overflow-hidden mb-6" style={{
          background: 'rgba(30, 41, 59, 0.4)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(71, 85, 105, 0.3)'
        }}>
          <div style={{ borderBottom: '1px solid rgba(71, 85, 105, 0.3)' }}>
            <ul className="flex overflow-x-auto">
              <li className="px-4 py-4 font-medium" style={{
                color: '#60A5FA',
                borderBottom: '2px solid #3B82F6'
              }}>
                Détails de configuration
              </li>
            </ul>
          </div>
          
          <div className="p-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center" style={{ color: '#FFFFFF' }}>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" style={{ color: '#60A5FA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                  </svg>
                  Configuration technique
                </h3>
                
                <div className="space-y-4">
                  <div className="rounded-lg p-4" style={{
                    background: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid rgba(59, 130, 246, 0.2)'
                  }}>
                    <h4 className="text-md font-medium mb-2" style={{ color: '#CBD5E1' }}>Chemin de l'exécutable</h4>
                    <div className="p-3 rounded-md font-mono text-sm overflow-x-auto" style={{
                      background: 'rgba(15, 23, 42, 0.5)',
                      border: '1px solid rgba(71, 85, 105, 0.4)',
                      color: '#94A3B8'
                    }}>
                      {workflow.executable_path || 'Non spécifié'}
                    </div>
                  </div>
                  
                  <div className="rounded-lg p-4" style={{
                    background: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid rgba(59, 130, 246, 0.2)'
                  }}>
                    <h4 className="text-md font-medium mb-2" style={{ color: '#CBD5E1' }}>Chemin des données d'entrée</h4>
                    <div className="p-3 rounded-md font-mono text-sm overflow-x-auto" style={{
                      background: 'rgba(15, 23, 42, 0.5)',
                      border: '1px solid rgba(71, 85, 105, 0.4)',
                      color: '#94A3B8'
                    }}>
                      {workflow.input_path || 'Non spécifié'}
                    </div>
                  </div>
                  
                  <div className="rounded-lg p-4" style={{
                    background: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid rgba(59, 130, 246, 0.2)'
                  }}>
                    <h4 className="text-md font-medium mb-2" style={{ color: '#CBD5E1' }}>Chemin des résultats</h4>
                    <div className="p-3 rounded-md font-mono text-sm overflow-x-auto" style={{
                      background: 'rgba(15, 23, 42, 0.5)',
                      border: '1px solid rgba(71, 85, 105, 0.4)',
                      color: '#94A3B8'
                    }}>
                      {workflow.output_path || 'Non spécifié'}
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center" style={{ color: '#FFFFFF' }}>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" style={{ color: '#60A5FA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  Chronologie d'exécution
                </h3>
                
                <div className="relative pl-8 space-y-6" style={{ borderLeft: '2px solid rgba(71, 85, 105, 0.3)' }}>
                  <div className="relative">
                    <div className="absolute -left-10 top-1 rounded-full w-6 h-6 flex items-center justify-center" style={{
                      background: '#3B82F6',
                      color: '#FFFFFF'
                    }}>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                      </svg>
                    </div>
                    <div className="p-4 rounded-lg" style={{
                      background: 'rgba(59, 130, 246, 0.1)',
                      border: '1px solid rgba(59, 130, 246, 0.2)'
                    }}>
                      <h4 className="text-md font-medium mb-1" style={{ color: '#CBD5E1' }}>Date de création</h4>
                      <p style={{ color: '#94A3B8' }}>{formatDate(workflow.created_at)}</p>
                    </div>
                  </div>
                  
                  <div className="relative">
                    <div className="absolute -left-10 top-1 rounded-full w-6 h-6 flex items-center justify-center" style={{
                      background: workflow.submitted_at ? '#3B82F6' : 'rgba(71, 85, 105, 0.5)',
                      color: '#FFFFFF'
                    }}>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                      </svg>
                    </div>
                    <div className="p-4 rounded-lg" style={{
                      background: workflow.submitted_at ? 'rgba(59, 130, 246, 0.1)' : 'rgba(71, 85, 105, 0.1)',
                      border: `1px solid ${workflow.submitted_at ? 'rgba(59, 130, 246, 0.2)' : 'rgba(71, 85, 105, 0.2)'}`
                    }}>
                      <h4 className="text-md font-medium mb-1" style={{ color: '#CBD5E1' }}>Date de soumission</h4>
                      <p style={{ color: '#94A3B8' }}>
                        {workflow.submitted_at ? formatDate(workflow.submitted_at) : 'Non soumis'}
                      </p>
                    </div>
                  </div>
                  
                  <div className="relative">
                    <div className="absolute -left-10 top-1 rounded-full w-6 h-6 flex items-center justify-center" style={{
                      background: workflow.completed_at ? '#10B981' : 'rgba(71, 85, 105, 0.5)',
                      color: '#FFFFFF'
                    }}>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div className="p-4 rounded-lg" style={{
                      background: workflow.completed_at ? 'rgba(16, 185, 129, 0.1)' : 'rgba(71, 85, 105, 0.1)',
                      border: `1px solid ${workflow.completed_at ? 'rgba(16, 185, 129, 0.2)' : 'rgba(71, 85, 105, 0.2)'}`
                    }}>
                      <h4 className="text-md font-medium mb-1" style={{ color: '#CBD5E1' }}>Date de complétion</h4>
                      <p style={{ color: '#94A3B8' }}>
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
        <div className="rounded-xl overflow-hidden mb-6" style={{
          background: 'rgba(30, 41, 59, 0.4)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(71, 85, 105, 0.3)'
        }}>
          <div className="px-6 py-4 flex justify-between items-center" style={{
            borderBottom: '1px solid rgba(71, 85, 105, 0.3)'
          }}>
            <h2 className="text-xl font-bold" style={{ color: '#FFFFFF' }}>Tâches</h2>
            {tasks.length > 0 && (
              <span className="text-sm font-medium px-3 py-1 rounded-full" style={{
                background: 'rgba(59, 130, 246, 0.15)',
                color: '#60A5FA',
                border: '1px solid rgba(59, 130, 246, 0.3)'
              }}>
                {tasks.filter(task => task.status === 'COMPLETED').length} / {tasks.length} terminées
              </span>
            )}
          </div>
          
          {tasks.length === 0 ? (
            <div className="text-center py-12 px-6">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto mb-4" style={{ color: '#60A5FA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              <h3 className="text-lg font-medium mb-2" style={{ color: '#FFFFFF' }}>
                {workflow.status === 'CREATED' 
                  ? 'Les tâches seront générées après la soumission du workflow.' 
                  : 'Aucune tâche trouvée pour ce workflow.'}
              </h3>
              <p className="max-w-md mx-auto" style={{ color: '#94A3B8' }}>
                {workflow.status === 'CREATED' 
                  ? 'Une fois le workflow soumis, les tâches associées apparaîtront ici.' 
                  : 'Vérifiez si le workflow a été configuré correctement.'}
              </p>
            </div>
          ) : (
            <div className="p-6">
              <div className="grid grid-cols-1 gap-4">
                {tasks.map((task) => {
                  const taskStatusInfo = getStatusInfo(task.status, 'task');
                  
                  return (
                    <div key={task.id} className="rounded-lg overflow-hidden transition-all duration-300" style={{
                      background: 'rgba(15, 23, 42, 0.5)',
                      border: '1px solid rgba(71, 85, 105, 0.4)'
                    }}>
                      <div className="p-4">
                        <div className="flex justify-between items-start">
                          <div className="flex items-center">
                            <div className="rounded-full w-10 h-10 flex items-center justify-center" style={{
                              background: task.status === 'COMPLETED'
                                ? 'rgba(16, 185, 129, 0.2)'
                                : task.status === 'RUNNING'
                                  ? 'rgba(59, 130, 246, 0.2)'
                                  : 'rgba(148, 163, 184, 0.2)',
                              border: `1px solid ${task.status === 'COMPLETED' ? '#10B981' : task.status === 'RUNNING' ? '#3B82F6' : '#94A3B8'}40`
                            }}>
                              {task.status === 'COMPLETED' ? (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" style={{ color: '#10B981' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                              ) : task.status === 'RUNNING' ? (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" style={{ color: '#3B82F6' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                </svg>
                              ) : (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" style={{ color: '#94A3B8' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                              )}
                            </div>
                            <div className="ml-4">
                              <h3 className="text-lg font-medium" style={{ color: '#FFFFFF' }}>{task.name}</h3>
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium mt-1" style={{
                                background: taskStatusInfo.bg,
                                color: taskStatusInfo.color,
                                border: `1px solid ${taskStatusInfo.color}40`
                              }}>
                                {taskStatusInfo.label}
                              </span>
                            </div>
                          </div>
                          <Link
                            href={`/workflows/${id}/tasks/${task.id}`}
                            className="text-sm font-medium flex items-center px-3 py-1 rounded-md transition-all duration-300" style={{
                              background: 'rgba(59, 130, 246, 0.2)',
                              color: '#60A5FA',
                              border: '1px solid rgba(59, 130, 246, 0.3)'
                            }}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                            </svg>
                            Détails
                          </Link>
                        </div>
                        
                        <div className="mt-4">
                          <div className="flex justify-between mb-1 text-sm font-medium" style={{ color: '#94A3B8' }}>
                            <span>Progression</span>
                            <span>{task.progress}%</span>
                          </div>
                          <div className="w-full rounded-full h-2.5 mb-4" style={{
                            background: 'rgba(71, 85, 105, 0.3)'
                          }}>
                            <div 
                              className="h-2.5 rounded-full transition-all duration-500"
                              style={{ 
                                width: `${task.progress}%`,
                                background: task.status === 'COMPLETED' 
                                  ? 'linear-gradient(90deg, #10B981 0%, #34D399 100%)' 
                                  : 'linear-gradient(90deg, #3B82F6 0%, #60A5FA 100%)'
                              }}
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
            className="flex items-center font-medium transition-all duration-300" style={{ color: '#60A5FA' }}
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
            </svg>
            Retour à la liste des workflows
          </Link>
          
          {['COMPLETED', 'PARTIAL_FAILURE', 'RUNNING', 'FAILED', 'AGGREGATING'].includes(workflow.status) && (
            <Link
              href={`/workflows/${workflow.id}/results`}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300" style={{
                background: 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
                border: '1px solid rgba(59, 130, 246, 0.4)',
                color: '#FFFFFF'
              }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Voir les résultats
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}