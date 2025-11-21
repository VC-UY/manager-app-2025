'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { workflowService } from '@/lib/api';

// Types des workflows
const workflowTypes = [
  { value: 'MATRIX_ADDITION', label: 'Addition de matrices de grande taille' },
  { value: 'MATRIX_MULTIPLICATION', label: 'Multiplication de matrices de grande taille' },
  { value: 'ML_TRAINING', label: 'Entraînement de modèle machine learning' },
  { value: 'OPEN_MALARIA', label: 'Simulation de propagation de la malaria'},
  { value: 'CUSTOM', label: 'Workflow personnalisé' }
];

export default function EditWorkflowPage() {
  const { id } = useParams();
  const router = useRouter();

  // État du formulaire
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    workflow_type: '',
    executable_path: '',
    input_path: '',
    output_path: '',
    priority: 1,
    max_execution_time: 3600,
    retry_count: 3
  });

  // États de l'interface
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [canEdit, setCanEdit] = useState(true);
  const [originalData, setOriginalData] = useState<any>(null);

  // Charger les données du workflow
  useEffect(() => {
    const fetchWorkflow = async () => {
      try {
        setLoading(true);
        const data = await workflowService.getWorkflow(id as string);
        setOriginalData(data);
        
        // Vérifier si le workflow est modifiable
        const nonEditableStates = ['SUBMITTED', 'RUNNING', 'COMPLETED', 'FAILED'];
        const isEditable = !nonEditableStates.includes(data.status);
        setCanEdit(isEditable);
        
        // Initialiser le formulaire avec les données existantes
        setFormData({
          name: data.name || '',
          description: data.description || '',
          workflow_type: data.workflow_type || 'CUSTOM',
          executable_path: data.executable_path || '',
          input_path: data.input_path || '',
          output_path: data.output_path || '',
          priority: data.priority || 1,
          max_execution_time: data.max_execution_time || 3600,
          retry_count: data.retry_count || 3
        });
        
        setError(null);
      } catch (err: any) {
        console.error('Erreur lors du chargement du workflow:', err);
        setError(err.error || 'Une erreur est survenue lors du chargement du workflow');
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchWorkflow();
    }
  }, [id]);

  // Gestion des changements de champs
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'priority' || name === 'max_execution_time' || name === 'retry_count' 
        ? parseInt(value, 10) 
        : value
    }));
  };

  // Soumission du formulaire
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!canEdit) {
      setError('Ce workflow ne peut plus être modifié car il a déjà été soumis ou est en cours d\'exécution.');
      return;
    }
    
    setSaving(true);
    setError(null);

    try {
      await workflowService.updateWorkflow(id as string, formData);
      router.push(`/workflows/${id}`);
    } catch (err: any) {
      console.error('Erreur lors de la mise à jour du workflow:', err);
      setError(err.error || 'Une erreur est survenue lors de la mise à jour du workflow');
    } finally {
      setSaving(false);
    }
  };

  // Obtenir le statut du workflow avec les couleurs et icône
  const getStatusInfo = (status: string) => {
    if (!originalData) return null;
    
    switch (status) {
      case 'CREATED':
        return {
          bgColor: 'rgba(100, 100, 100, 0.2)',
          textColor: '#A0A0A0',
          borderColor: 'rgba(150, 150, 150, 0.3)',
          label: 'Créé',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          )
        };
      case 'VALIDATED':
        return {
          bgColor: 'rgba(0, 180, 240, 0.2)',
          textColor: '#00D4FF',
          borderColor: 'rgba(0, 212, 255, 0.4)',
          label: 'Validé',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )
        };
      case 'SUBMITTED':
        return {
          bgColor: 'rgba(255, 165, 0, 0.2)',
          textColor: '#FFA500',
          borderColor: 'rgba(255, 165, 0, 0.4)',
          label: 'Soumis',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          )
        };
      case 'RUNNING':
        return {
          bgColor: 'rgba(0, 255, 0, 0.2)',
          textColor: '#00FF00',
          borderColor: 'rgba(0, 255, 0, 0.4)',
          label: 'En cours',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          )
        };
      case 'COMPLETED':
        return {
          bgColor: 'rgba(0, 255, 0, 0.3)',
          textColor: '#00FF00',
          borderColor: 'rgba(0, 255, 0, 0.5)',
          label: 'Terminé',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )
        };
      case 'FAILED':
        return {
          bgColor: 'rgba(255, 0, 0, 0.2)',
          textColor: '#FF0000',
          borderColor: 'rgba(255, 0, 0, 0.4)',
          label: 'Échoué',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )
        };
      default:
        return null;
    }
  };

  const formattedDate = (dateString: string) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('fr-FR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  if (loading) {
    return (
      <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #001440 0%, #002060 50%, #001440 100%)' }}>
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col justify-center items-center h-64 backdrop-blur-xl rounded-xl"
            style={{
              background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.8) 0%, rgba(0, 20, 64, 0.8) 100%)',
              border: '2px solid rgba(0, 180, 240, 0.3)',
            }}>
            <div className="relative mb-4">
              <div className="animate-spin rounded-full h-16 w-16 border-4 border-slate-600"></div>
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 absolute top-0" style={{ borderTopColor: '#00D4FF' }}></div>
            </div>
            <p className="text-lg font-medium animate-pulse" style={{ color: '#00D4FF' }}>Chargement du formulaire d'édition...</p>
          </div>
        </div>
      </div>
    );
  }

  const statusInfo = originalData && getStatusInfo(originalData.status);

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #001440 0%, #002060 50%, #001440 100%)' }}>
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Bannière en haut */}
        <div className="rounded-3xl mb-8 overflow-hidden relative backdrop-blur-xl"
          style={{
            background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.9) 0%, rgba(0, 20, 64, 0.9) 100%)',
            border: '2px solid rgba(0, 180, 240, 0.3)',
            boxShadow: '0 8px 32px rgba(0, 32, 96, 0.5)',
          }}>
          <div className="px-8 py-6">
            <div className="flex flex-col md:flex-row md:justify-between md:items-center space-y-4 md:space-y-0">
              <div>
                <div className="flex items-center">
                  <Link
                    href={`/workflows/${id}`}
                    className="mr-3 p-2 rounded-full transition-all duration-200"
                    style={{
                      background: 'rgba(0, 180, 240, 0.2)',
                      border: '1px solid rgba(0, 180, 240, 0.3)',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(0, 212, 255, 0.3)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'rgba(0, 180, 240, 0.2)';
                    }}>
                    <svg className="w-5 h-5" style={{ color: '#00D4FF' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                    </svg>
                  </Link>
                  <h1 className="text-2xl md:text-3xl font-bold"
                    style={{
                      background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      letterSpacing: '0.5px',
                    }}>Modifier le workflow</h1>
                </div>
                {originalData && (
                  <div className="flex flex-wrap items-center mt-3 gap-2">
                    {statusInfo && (
                      <span className="flex items-center px-3 py-1 text-sm font-medium rounded-full backdrop-blur-sm border"
                        style={{
                          background: statusInfo.bgColor,
                          color: statusInfo.textColor,
                          borderColor: statusInfo.borderColor,
                        }}>
                        {statusInfo.icon}
                        {statusInfo.label}
                      </span>
                    )}
                    <span className="px-3 py-1 text-sm rounded-full backdrop-blur-sm inline-flex items-center"
                      style={{
                        background: 'rgba(0, 180, 240, 0.2)',
                        color: '#00B0F0',
                        border: '1px solid rgba(0, 180, 240, 0.3)',
                      }}>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      Créé le {formattedDate(originalData.created_at)}
                    </span>
                  </div>
                )}
              </div>
              <div className="flex items-center">
                {!canEdit && (
                  <div className="px-4 py-2 rounded-lg text-sm font-medium backdrop-blur-sm flex items-center border"
                    style={{
                      background: 'rgba(255, 165, 0, 0.2)',
                      color: '#FFA500',
                      borderColor: 'rgba(255, 165, 0, 0.3)',
                    }}>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    Lecture seule
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        
        {error && (
          <div className="border-l-4 p-4 mb-6 rounded-lg backdrop-blur-xl"
            style={{
              background: 'linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, rgba(185, 28, 28, 0.05) 100%)',
              borderColor: '#F87171',
              boxShadow: '0 4px 16px rgba(220, 38, 38, 0.2)',
            }}>
            <div className="flex items-center">
              <svg className="h-5 w-5 mr-2" style={{ color: '#F87171' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-medium" style={{ color: '#F87171' }}>{error}</span>
            </div>
          </div>
        )}

        {!canEdit && (
          <div className="border-l-4 p-4 mb-6 rounded-lg backdrop-blur-xl"
            style={{
              background: 'linear-gradient(135deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 140, 0, 0.05) 100%)',
              borderColor: '#FFA500',
              boxShadow: '0 4px 16px rgba(255, 165, 0, 0.2)',
            }}>
            <div className="flex items-center">
              <svg className="h-5 w-5 mr-2" style={{ color: '#FFA500' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span className="font-medium" style={{ color: '#FFA500' }}>Ce workflow est en lecture seule car il a déjà été soumis ou est en cours d'exécution.</span>
            </div>
          </div>
        )}

        <div className="rounded-3xl p-6 border backdrop-blur-xl"
          style={{
            background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.8) 0%, rgba(0, 20, 64, 0.8) 100%)',
            borderColor: 'rgba(0, 180, 240, 0.3)',
            boxShadow: '0 8px 32px rgba(0, 32, 96, 0.5)',
          }}>
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Informations de base */}
            <div>
              <div className="flex items-center mb-4">
                <div className="p-2 rounded-xl mr-3"
                  style={{
                    background: 'linear-gradient(135deg, rgba(0, 180, 240, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%)',
                    border: '2px solid rgba(0, 180, 240, 0.3)',
                  }}>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" style={{ color: '#00D4FF' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>Informations de base</h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                <div>
                  <label htmlFor="name" className="block text-md font-medium mb-2" style={{ color: '#00B0F0' }}>
                    Nom du workflow *
                  </label>
                  <input
                    id="name"
                    name="name"
                    type="text"
                    required
                    value={formData.name}
                    onChange={handleChange}
                    disabled={!canEdit}
                    className="block w-full border rounded-xl py-3 px-4 focus:outline-none focus:ring-2 transition-all duration-200"
                    style={{
                      background: canEdit ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.2)',
                      borderColor: 'rgba(0, 180, 240, 0.3)',
                      color: '#FFFFFF',
                    }}
                    placeholder="Nom du workflow"
                    onFocus={(e) => {
                      if (canEdit) {
                        e.currentTarget.style.borderColor = '#00D4FF';
                        e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.1)';
                      }
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  />
                </div>
                <div>
                  <label htmlFor="workflow_type" className="block text-md font-medium mb-2" style={{ color: '#00B0F0' }}>
                    Type de workflow *
                  </label>
                  <div className="relative">
                    <select
                      id="workflow_type"
                      name="workflow_type"
                      required
                      value={formData.workflow_type}
                      onChange={handleChange}
                      disabled={!canEdit}
                      className="block w-full rounded-xl border py-3 px-4 pr-10 appearance-none focus:outline-none focus:ring-2 transition-all duration-200"
                      style={{
                        background: canEdit ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.2)',
                        borderColor: 'rgba(0, 180, 240, 0.3)',
                        color: '#FFFFFF',
                      }}
                      onFocus={(e) => {
                        if (canEdit) {
                          e.currentTarget.style.borderColor = '#00D4FF';
                          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.1)';
                        }
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}>
                      {workflowTypes.map(type => (
                        <option key={type.value} value={type.value} style={{ background: '#001440', color: '#FFFFFF' }}>
                          {type.label}
                        </option>
                      ))}
                    </select>
                    <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                      <svg className="h-5 w-5" style={{ color: '#00B0F0' }} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
              <div className="mt-6">
                <label htmlFor="description" className="block text-md font-medium mb-2" style={{ color: '#00B0F0' }}>
                  Description
                </label>
                <textarea
                  id="description"
                  name="description"
                  rows={4}
                  value={formData.description}
                  onChange={handleChange}
                  disabled={!canEdit}
                  placeholder="Décrivez l'objectif et les caractéristiques de ce workflow"
                  className="block w-full border rounded-xl py-3 px-4 focus:outline-none focus:ring-2 transition-all duration-200"
                  style={{
                    background: canEdit ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.2)',
                    borderColor: 'rgba(0, 180, 240, 0.3)',
                    color: '#FFFFFF',
                  }}
                  onFocus={(e) => {
                    if (canEdit) {
                      e.currentTarget.style.borderColor = '#00D4FF';
                      e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.1)';
                    }
                  }}
                  onBlur={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                />
              </div>
            </div>

            {/* Paramètres d'exécution */}
            <div>
              <div className="flex items-center mb-4">
                <div className="p-2 rounded-xl mr-3"
                  style={{
                    background: 'linear-gradient(135deg, rgba(0, 255, 0, 0.2) 0%, rgba(0, 200, 0, 0.1) 100%)',
                    border: '2px solid rgba(0, 255, 0, 0.3)',
                  }}>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" style={{ color: '#00FF00' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>Paramètres d'exécution</h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4 p-5 rounded-xl border"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 180, 240, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%)',
                  borderColor: 'rgba(0, 180, 240, 0.2)',
                }}>
                <div>
                  <label htmlFor="executable_path" className="block text-md font-medium mb-2" style={{ color: '#00B0F0' }}>
                    Chemin de l'exécutable
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" style={{ color: '#00D4FF' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                    </div>
                    <input
                      id="executable_path"
                      name="executable_path"
                      type="text"
                      value={formData.executable_path}
                      onChange={handleChange}
                      disabled={!canEdit}
                      placeholder="/chemin/vers/executable.py"
                      className="block w-full border rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 transition-all duration-200"
                      style={{
                        background: canEdit ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.2)',
                        borderColor: 'rgba(0, 180, 240, 0.3)',
                        color: '#FFFFFF',
                      }}
                      onFocus={(e) => {
                        if (canEdit) {
                          e.currentTarget.style.borderColor = '#00D4FF';
                          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.1)';
                        }
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>
                </div>
                
                <div>
                  <label htmlFor="input_path" className="block text-md font-medium mb-2" style={{ color: '#00B0F0' }}>
                    Chemin des données d'entrée
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" style={{ color: '#00D4FF' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                      </svg>
                    </div>
                    <input
                      id="input_path"
                      name="input_path"
                      type="text"
                      value={formData.input_path}
                      onChange={handleChange}
                      disabled={!canEdit}
                      placeholder="/chemin/vers/données.csv"
                      className="block w-full border rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 transition-all duration-200"
                      style={{
                        background: canEdit ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.2)',
                        borderColor: 'rgba(0, 180, 240, 0.3)',
                        color: '#FFFFFF',
                      }}
                      onFocus={(e) => {
                        if (canEdit) {
                          e.currentTarget.style.borderColor = '#00D4FF';
                          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.1)';
                        }
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>
                </div>
                
                <div>
                  <label htmlFor="output_path" className="block text-md font-medium mb-2" style={{ color: '#00B0F0' }}>
                    Chemin des résultats
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" style={{ color: '#00D4FF' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <input
                      id="output_path"
                      name="output_path"
                      type="text"
                      value={formData.output_path}
                      onChange={handleChange}
                      disabled={!canEdit}
                      placeholder="/chemin/vers/résultats/"
                      className="block w-full border rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 transition-all duration-200"
                      style={{
                        background: canEdit ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.2)',
                        borderColor: 'rgba(0, 180, 240, 0.3)',
                        color: '#FFFFFF',
                      }}
                      onFocus={(e) => {
                        if (canEdit) {
                          e.currentTarget.style.borderColor = '#00D4FF';
                          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.1)';
                        }
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Paramètres avancés */}
            <div>
              <div className="flex items-center mb-4">
                <div className="p-2 rounded-xl mr-3"
                  style={{
                    background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.2) 0%, rgba(126, 34, 206, 0.1) 100%)',
                    border: '2px solid rgba(147, 51, 234, 0.3)',
                  }}>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" style={{ color: '#A78BFA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>Paramètres avancés</h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4 p-5 rounded-xl border"
                style={{
                  background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.1) 0%, rgba(126, 34, 206, 0.05) 100%)',
                  borderColor: 'rgba(147, 51, 234, 0.3)',
                }}>
                <div>
                  <label htmlFor="priority" className="block text-md font-medium mb-2" style={{ color: '#00B0F0' }}>
                    Priorité (1-10)
                  </label>
                  <div className="relative">
                    <input
                      id="priority"
                      name="priority"
                      type="number"
                      min="1"
                      max="10"
                      value={formData.priority}
                      onChange={handleChange}
                      disabled={!canEdit}
                      className="block w-full border rounded-xl py-3 px-4 focus:outline-none focus:ring-2 transition-all duration-200"
                      style={{
                        background: canEdit ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.2)',
                        borderColor: 'rgba(147, 51, 234, 0.3)',
                        color: '#FFFFFF',
                      }}
                      onFocus={(e) => {
                        if (canEdit) {
                          e.currentTarget.style.borderColor = '#A78BFA';
                          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(147, 51, 234, 0.1)';
                        }
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(147, 51, 234, 0.3)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                    <div className="absolute -bottom-6 left-0 right-0 mt-1">
                      <div className="flex justify-between items-center text-xs" style={{ color: '#A78BFA' }}>
                        <span>Basse</span>
                        <span className="font-medium">
                          {formData.priority <= 3 ? 'Basse' : formData.priority <= 7 ? 'Moyenne' : 'Haute'}
                        </span>
                        <span>Haute</span>
                      </div>
                      <div className="w-full rounded-full h-1 mt-1" style={{ background: 'rgba(147, 51, 234, 0.3)' }}>
                        <div 
                          className="h-1 rounded-full transition-all duration-300" 
                          style={{ 
                            width: `${formData.priority * 10}%`,
                            background: 'linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%)',
                          }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="pt-6 md:pt-0">
                  <label htmlFor="max_execution_time" className="block text-md font-medium mb-2" style={{ color: '#00B0F0' }}>
                    Temps d'exécution max (s)
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" style={{ color: '#A78BFA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <input
                      id="max_execution_time"
                      name="max_execution_time"
                      type="number"
                      min="60"
                      value={formData.max_execution_time}
                      onChange={handleChange}
                      disabled={!canEdit}
                      className="block w-full border rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 transition-all duration-200"
                      style={{
                        background: canEdit ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.2)',
                        borderColor: 'rgba(147, 51, 234, 0.3)',
                        color: '#FFFFFF',
                      }}
                      onFocus={(e) => {
                        if (canEdit) {
                          e.currentTarget.style.borderColor = '#A78BFA';
                          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(147, 51, 234, 0.1)';
                        }
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(147, 51, 234, 0.3)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                    <div className="absolute -bottom-6 left-0 text-xs" style={{ color: '#A78BFA' }}>
                      {Math.floor(formData.max_execution_time / 3600) > 0 
                        ? `${Math.floor(formData.max_execution_time / 3600)}h ${Math.floor((formData.max_execution_time % 3600) / 60)}m`
                        : `${Math.floor(formData.max_execution_time / 60)}m ${formData.max_execution_time % 60}s`
                      }
                    </div>
                  </div>
                </div>
                
                <div className="pt-6 md:pt-0">
                  <label htmlFor="retry_count" className="block text-md font-medium mb-2" style={{ color: '#00B0F0' }}>
                    Nombre de tentatives
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" style={{ color: '#A78BFA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                    </div>
                    <input
                      id="retry_count"
                      name="retry_count"
                      type="number"
                      min="0"
                      max="10"
                      value={formData.retry_count}
                      onChange={handleChange}
                      disabled={!canEdit}
                      className="block w-full border rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 transition-all duration-200"
                      style={{
                        background: canEdit ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.2)',
                        borderColor: 'rgba(147, 51, 234, 0.3)',
                        color: '#FFFFFF',
                      }}
                      onFocus={(e) => {
                        if (canEdit) {
                          e.currentTarget.style.borderColor = '#A78BFA';
                          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(147, 51, 234, 0.1)';
                        }
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(147, 51, 234, 0.3)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Aide contextuelle */}
            <div className="p-4 rounded-xl border flex items-start backdrop-blur-sm"
              style={{
                background: 'linear-gradient(135deg, rgba(0, 180, 240, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%)',
                borderColor: 'rgba(0, 180, 240, 0.3)',
              }}>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mt-0.5 mr-3 flex-shrink-0" style={{ color: '#00D4FF' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <h3 className="text-md font-medium mb-1" style={{ color: '#00D4FF' }}>Conseils pour les paramètres</h3>
                <ul className="text-sm space-y-1 list-disc list-inside" style={{ color: '#00B0F0' }}>
                  <li>Choisissez une priorité plus élevée pour les workflows urgents</li>
                  <li>Le temps d'exécution maximum doit être suffisant pour permettre l'achèvement du workflow</li>
                  <li>Définissez plusieurs tentatives pour les workflows avec des dépendances externes</li>
                </ul>
              </div>
            </div>

            {/* Boutons d'action */}
            <div className="flex flex-col-reverse sm:flex-row justify-between items-center pt-6"
              style={{ borderTop: '1px solid rgba(0, 180, 240, 0.2)' }}>
              <Link
                href={`/workflows/${id}`}
                className="mt-3 sm:mt-0 py-2.5 px-5 border rounded-xl font-medium focus:outline-none transition-all duration-200 w-full sm:w-auto text-center"
                style={{
                  borderColor: 'rgba(0, 180, 240, 0.3)',
                  color: '#00B0F0',
                  background: 'rgba(0, 180, 240, 0.1)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(0, 180, 240, 0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(0, 180, 240, 0.1)';
                }}>
                Retour aux détails
              </Link>
              <div className="flex gap-3 w-full sm:w-auto">
                <Link
                  href={`/workflows/${id}`}
                  className="py-2.5 px-5 border rounded-xl font-medium focus:outline-none transition-all duration-200 w-full sm:w-auto text-center"
                  style={{
                    borderColor: 'rgba(100, 100, 100, 0.3)',
                    color: '#A0A0A0',
                    background: 'rgba(100, 100, 100, 0.1)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(100, 100, 100, 0.2)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'rgba(100, 100, 100, 0.1)';
                  }}>
                  Annuler
                </Link>
                {canEdit && (
                  <button
                    type="submit"
                    disabled={saving || !canEdit}
                    className="py-2.5 px-5 border rounded-xl font-medium focus:outline-none transition-all duration-200 w-full sm:w-auto"
                    style={{
                      background: saving || !canEdit 
                        ? 'rgba(0, 180, 240, 0.3)' 
                        : 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                      color: '#FFFFFF',
                      borderColor: 'rgba(0, 212, 255, 0.4)',
                      boxShadow: saving || !canEdit ? 'none' : '0 4px 16px rgba(0, 180, 240, 0.3)',
                      cursor: saving || !canEdit ? 'not-allowed' : 'pointer',
                    }}
                    onMouseEnter={(e) => {
                      if (!saving && canEdit) {
                        e.currentTarget.style.transform = 'translateY(-2px)';
                        e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 212, 255, 0.5)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!saving && canEdit) {
                        e.currentTarget.style.transform = 'translateY(0)';
                        e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 180, 240, 0.3)';
                      }
                    }}>
                    {saving ? (
                      <div className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Enregistrement...
                      </div>
                    ) : (
                      'Enregistrer les modifications'
                    )}
                  </button>
                )}
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}