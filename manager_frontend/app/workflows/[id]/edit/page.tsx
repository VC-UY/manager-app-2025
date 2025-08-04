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
  { value: 'OPEN_MALARIA', label: 'Simulation de propagation de la malaria', icon: '🦟' },
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
      // Mettre à jour le workflow
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
        return null;
    }
  };

  // Obtenir l'icône pour le type de workflow
  const getWorkflowTypeIcon = (type: string) => {
    switch (type) {
      case 'MATRIX_ADDITION':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
        );
      case 'MATRIX_MULTIPLICATION':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v1m6 11h2m-6 0h-2v4m0-11v3m0 0h.01M12 12h4.01M16 20h4M4 12h4m12 0h.01" />
          </svg>
        );
      case 'ML_TRAINING':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        );
      case 'CUSTOM':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        );
      default:
        return (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        );
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
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col justify-center items-center h-64 bg-white rounded-xl shadow-md">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500 mb-4"></div>
          <p className="text-blue-800 text-lg font-medium animate-pulse">Chargement du formulaire d'édition...</p>
        </div>
      </div>
    );
  }

  const statusInfo = originalData && getStatusInfo(originalData.status);

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      {/* Bannière en haut */}
      <div className="bg-gradient-to-r from-blue-700 to-indigo-800 rounded-xl shadow-lg mb-8 overflow-hidden relative">
        <div className="absolute inset-0 bg-grid-white/10 opacity-10"></div>
        <div className="relative z-10 px-8 py-6 text-white">
          <div className="flex flex-col md:flex-row md:justify-between md:items-center space-y-4 md:space-y-0">
            <div>
              <div className="flex items-center">
                <Link
                  href={`/workflows/${id}`}
                  className="mr-3 bg-white/10 hover:bg-white/20 p-2 rounded-full transition-colors duration-200"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                  </svg>
                </Link>
                <h1 className="text-2xl md:text-3xl font-bold">Modifier le workflow</h1>
              </div>
              {originalData && (
                <div className="flex flex-wrap items-center mt-3 gap-2">
                  {statusInfo && (
                    <span className={`flex items-center px-3 py-1 text-sm font-medium rounded-full ${statusInfo.bgColor} ${statusInfo.textColor} border ${statusInfo.borderColor}`}>
                      {statusInfo.icon}
                      {statusInfo.label}
                    </span>
                  )}
                  <span className="bg-white/10 text-white px-3 py-1 text-sm rounded-full backdrop-blur-sm inline-flex items-center">
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
                <div className="bg-white/10 text-white px-4 py-2 rounded-lg text-sm font-medium backdrop-blur-sm flex items-center border border-yellow-300/30">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1.5 text-yellow-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  Lecture seule
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Vagues décoratives */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320">
            <path fill="rgba(255,255,255,0.1)" fillOpacity="1" d="M0,288L48,272C96,256,192,224,288,197.3C384,171,480,149,576,165.3C672,181,768,235,864,250.7C960,267,1056,245,1152,224C1248,203,1344,181,1392,170.7L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
          </svg>
        </div>
      </div>
      
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 text-red-800 p-4 mb-6 rounded-lg shadow-sm">
          <div className="flex items-center">
            <svg className="h-5 w-5 text-red-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="font-medium">{error}</span>
          </div>
        </div>
      )}

      {!canEdit && (
        <div className="bg-amber-50 border-l-4 border-amber-500 text-amber-800 p-4 mb-6 rounded-lg shadow-sm">
          <div className="flex items-center">
            <svg className="h-5 w-5 text-amber-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="font-medium">Ce workflow est en lecture seule car il a déjà été soumis ou est en cours d'exécution.</span>
          </div>
        </div>
      )}

      <div className="bg-white shadow-md rounded-xl p-6 border border-gray-100">
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Informations de base */}
          <div>
            <div className="flex items-center mb-4">
              <div className="p-2 rounded-full bg-blue-100 mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="text-xl font-bold text-blue-900">Informations de base</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
              <div>
                <label htmlFor="name" className="block text-md font-medium text-blue-800 mb-2">
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
                  className={`block w-full border rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-blue-500 text-blue-900 ${
                    !canEdit 
                      ? 'bg-blue-50 border-blue-200 cursor-not-allowed' 
                      : 'border-blue-300 hover:border-blue-400 focus:border-blue-500'
                  }`}
                  placeholder="Nom du workflow"
                />
              </div>
              <div>
                <label htmlFor="workflow_type" className="block text-md font-medium text-blue-800 mb-2">
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
                    className={`block w-full rounded-lg border shadow-sm py-3 px-4 pr-10 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500 text-blue-900 ${
                      !canEdit 
                        ? 'bg-blue-50 border-blue-200 cursor-not-allowed' 
                        : 'border-blue-300 hover:border-blue-400 focus:border-blue-500'
                    }`}
                  >
                    {workflowTypes.map(type => (
                      <option key={type.value} value={type.value}>
                        {type.label}
                      </option>
                    ))}
                  </select>
                  <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                    <svg className="h-5 w-5 text-blue-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-6">
              <label htmlFor="description" className="block text-md font-medium text-blue-800 mb-2">
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
                className={`block w-full border rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-blue-500 text-blue-900 ${
                  !canEdit 
                    ? 'bg-blue-50 border-blue-200 cursor-not-allowed' 
                    : 'border-blue-300 hover:border-blue-400 focus:border-blue-500'
                }`}
              />
            </div>
          </div>

          {/* Paramètres d'exécution */}
          <div>
            <div className="flex items-center mb-4">
              <div className="p-2 rounded-full bg-green-100 mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <h2 className="text-xl font-bold text-blue-900">Paramètres d'exécution</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4 bg-blue-50 p-5 rounded-lg border border-blue-100">
              <div>
                <label htmlFor="executable_path" className="block text-md font-medium text-blue-800 mb-2">
                  Chemin de l'exécutable
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
                    className={`block w-full border rounded-lg shadow-sm py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-blue-500 text-blue-900 ${
                      !canEdit 
                        ? 'bg-blue-50/50 border-blue-200 cursor-not-allowed' 
                        : 'bg-white border-blue-300 hover:border-blue-400 focus:border-blue-500'
                    }`}
                  />
                </div>
              </div>
              
              <div>
                <label htmlFor="input_path" className="block text-md font-medium text-blue-800 mb-2">
                  Chemin des données d'entrée
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
                    className={`block w-full border rounded-lg shadow-sm py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-blue-500 text-blue-900 ${
                      !canEdit 
                        ? 'bg-blue-50/50 border-blue-200 cursor-not-allowed' 
                        : 'bg-white border-blue-300 hover:border-blue-400 focus:border-blue-500'
                    }`}
                  />
                </div>
              </div>
              
              <div>
                <label htmlFor="output_path" className="block text-md font-medium text-blue-800 mb-2">
                  Chemin des résultats
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
                    className={`block w-full border rounded-lg shadow-sm py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-blue-500 text-blue-900 ${
                      !canEdit 
                        ? 'bg-blue-50/50 border-blue-200 cursor-not-allowed' 
                        : 'bg-white border-blue-300 hover:border-blue-400 focus:border-blue-500'
                    }`}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Paramètres avancés */}
          <div>
            <div className="flex items-center mb-4">
              <div className="p-2 rounded-full bg-purple-100 mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </div>
              <h2 className="text-xl font-bold text-blue-900">Paramètres avancés</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4 p-5 bg-purple-50 rounded-lg border border-purple-100">
              <div>
                <label htmlFor="priority" className="block text-md font-medium text-blue-800 mb-2">
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
                    className={`block w-full border rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-purple-500 text-blue-900 ${
                      !canEdit 
                        ? 'bg-purple-50/50 border-purple-200 cursor-not-allowed' 
                        : 'bg-white border-purple-300 hover:border-purple-400 focus:border-purple-500'
                    }`}
                  />
                  <div className="absolute -bottom-6 left-0 right-0 mt-1">
                    <div className="flex justify-between items-center text-xs text-purple-700">
                      <span>Basse</span>
                      <span className="font-medium">
                        {formData.priority <= 3 ? 'Basse' : formData.priority <= 7 ? 'Moyenne' : 'Haute'}
                      </span>
                      <span>Haute</span>
                    </div>
                    <div className="w-full bg-purple-200 rounded-full h-1 mt-1">
                      <div 
                        className="bg-purple-600 h-1 rounded-full" 
                        style={{ width: `${formData.priority * 10}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="pt-6 md:pt-0">
                <label htmlFor="max_execution_time" className="block text-md font-medium text-blue-800 mb-2">
                  Temps d'exécution max (s)
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
                    className={`block w-full border rounded-lg shadow-sm py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-purple-500 text-blue-900 ${
                      !canEdit 
                        ? 'bg-purple-50/50 border-purple-200 cursor-not-allowed' 
                        : 'bg-white border-purple-300 hover:border-purple-400 focus:border-purple-500'
                    }`}
                  />
                  <div className="absolute -bottom-6 left-0 text-xs text-purple-700">
                    {Math.floor(formData.max_execution_time / 3600) > 0 
                      ? `${Math.floor(formData.max_execution_time / 3600)}h ${Math.floor((formData.max_execution_time % 3600) / 60)}m`
                      : `${Math.floor(formData.max_execution_time / 60)}m ${formData.max_execution_time % 60}s`
                    }
                  </div>
                </div>
              </div>
              
              <div className="pt-6 md:pt-0">
                <label htmlFor="retry_count" className="block text-md font-medium text-blue-800 mb-2">
                  Nombre de tentatives
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
                    className={`block w-full border rounded-lg shadow-sm py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-purple-500 text-blue-900 ${
                      !canEdit 
                        ? 'bg-purple-50/50 border-purple-200 cursor-not-allowed' 
                        : 'bg-white border-purple-300 hover:border-purple-400 focus:border-purple-500'
                    }`}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Aide contextuelle */}
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 flex items-start">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-500 mt-0.5 mr-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <h3 className="text-md font-medium text-blue-800 mb-1">Conseils pour les paramètres</h3>
              <ul className="text-sm text-blue-700 space-y-1 list-disc list-inside">
                <li>Choisissez une priorité plus élevée pour les workflows urgents</li>
                <li>Le temps d'exécution maximum doit être suffisant pour permettre l'achèvement du workflow</li>
                <li>Définissez plusieurs tentatives pour les workflows avec des dépendances externes</li>
              </ul>
            </div>
          </div>

          {/* Boutons d'action */}
          <div className="flex flex-col-reverse sm:flex-row justify-between items-center pt-6 border-t border-gray-200">
            <Link
              href={`/workflows/${id}`}
              className="mt-3 sm:mt-0 py-2.5 px-5 border border-blue-300 rounded-lg text-blue-700 bg-white hover:bg-blue-50 font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200 w-full sm:w-auto text-center"
            >
              Retour aux détails
            </Link>
            <div className="flex gap-3 w-full sm:w-auto">
              <Link
                href={`/workflows/${id}`}
                className="py-2.5 px-5 border border-gray-300 rounded-lg shadow-sm text-gray-700 bg-white hover:bg-gray-50 font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200 w-full sm:w-auto text-center"
              >
                Annuler
              </Link>
              {canEdit && (
                <button
                  type="submit"
                  disabled={saving || !canEdit}
                  className={`py-2.5 px-5 border border-transparent rounded-lg shadow-sm text-white bg-blue-600 hover:bg-blue-700 font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200 w-full sm:w-auto ${
                    saving || !canEdit ? 'opacity-70 cursor-not-allowed' : ''
                  }`}
                >
                  {saving ? (
                    <div className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
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
  );
}