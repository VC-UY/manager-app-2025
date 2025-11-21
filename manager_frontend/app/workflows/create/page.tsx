
'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { workflowService } from '@/lib/api';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { HiOutlineDocumentAdd, HiOutlineCog, HiOutlineClock, HiOutlineArrowLeft, HiOutlineCalculator, HiOutlineCube, HiOutlineChip, HiOutlineBeaker } from 'react-icons/hi';

// Types des workflows avec des icônes associées
const workflowTypes = [
  { value: 'MATRIX_ADDITION', label: 'Addition de matrices de grande taille', icon: HiOutlineCalculator },
  { value: 'MATRIX_MULTIPLICATION', label: 'Multiplication de matrices de grande taille', icon: HiOutlineCube },
  { value: 'ML_TRAINING', label: 'Entraînement de modèle machine learning', icon: HiOutlineChip },
  { value: 'OPEN_MALARIA', label: 'Simulation de propagation de la malaria', icon: HiOutlineBeaker },
{ value: 'CUSTOM', label: 'Workflow personnalisé', icon: HiOutlineCog }

];

export default function CreateWorkflowPage() {
  const router = useRouter();

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    workflow_type: 'ML_TRAINING',
    executable_path: '',
    input_path: '',
    output_path: '',
    priority: 1,
    max_execution_time: 3600,
    retry_count: 3
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    
    if (name === 'priority' || name === 'max_execution_time' || name === 'retry_count') {
      const numValue = parseInt(value, 10);
      setFormData(prev => ({
        ...prev,
        [name]: isNaN(numValue) ? 0 : numValue
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  const handleSelectPath = async (fieldName: string): Promise<void> => {
    try {
      const hasFileSystemAccess = 'showOpenFilePicker' in window || 'showDirectoryPicker' in window;
      
      if (hasFileSystemAccess) {
        if (fieldName === 'executable_path') {
          const showOpenFilePicker = (window as any).showOpenFilePicker;
          if (showOpenFilePicker) {
            const fileHandle = await showOpenFilePicker({
              types: [
                {
                  description: 'Fichiers exécutables',
                  accept: {
                    'text/x-python': ['.py'],
                    'application/x-sh': ['.sh'],
                    'application/x-msdos-program': ['.exe', '.bat']
                  }
                }
              ],
              multiple: false
            });
            
            if (fileHandle && fileHandle[0]) {
              setFormData(prev => ({
                ...prev,
                [fieldName]: fileHandle[0].name
              }));
              return;
            }
          }
        } else {
          const showDirectoryPicker = (window as any).showDirectoryPicker;
          if (showDirectoryPicker) {
            const directoryHandle = await showDirectoryPicker({
              mode: 'readwrite'
            });
            
            if (directoryHandle) {
              setFormData(prev => ({
                ...prev,
                [fieldName]: directoryHandle.name
              }));
              return;
            }
          }
        }
      }
      
      useFallbackFilePicker(fieldName);
      
    } catch (error: unknown) {
      const err = error as { name?: string };
      if (err.name !== 'AbortError') {
        console.error('Erreur lors de la sélection du chemin:', error);
        useFallbackFilePicker(fieldName);
      }
    }
  };

  const useFallbackFilePicker = (fieldName: string): void => {
    const input = document.createElement('input');
    
    if (fieldName === 'executable_path') {
      input.type = 'file';
      input.accept = '.py,.exe,.sh,.bat';
    } else {
      input.type = 'file';
      (input as any).webkitdirectory = true;
    }
    
    input.onchange = (event: Event) => {
      const target = event.target as HTMLInputElement;
      const files = target.files;
      
      if (files && files.length > 0) {
        if (fieldName === 'executable_path') {
          setFormData(prev => ({
            ...prev,
            [fieldName]: files[0].name
          }));
        } else {
          const file = files[0] as any;
          const path = file.webkitRelativePath || file.name;
          const folderPath = path.includes('/') ? path.substring(0, path.indexOf('/')) : path;
          setFormData(prev => ({
            ...prev,
            [fieldName]: folderPath
          }));
        }
      }
    };
    
    input.click();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const dataToSubmit = {
        ...formData,
        priority: isNaN(formData.priority) ? 1 : formData.priority,
        max_execution_time: isNaN(formData.max_execution_time) ? 3600 : formData.max_execution_time,
        retry_count: isNaN(formData.retry_count) ? 3 : formData.retry_count
      };

      const response = await workflowService.createWorkflow(dataToSubmit);
      router.push(`/workflows/${response.id}`);
    } catch (err: any) {
      console.error('Erreur lors de la création du workflow:', err);
      setError(err.error ?? 'Une erreur est survenue lors de la création du workflow');
    } finally {
      setLoading(false);
    }
  };

  interface WorkflowType {
    value: string;
    label: string;
    icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
  }

  const getWorkflowIcon = (type: string) => {
    const workflowType: WorkflowType | undefined = workflowTypes.find(wt => wt.value === type);
    return workflowType ? workflowType.icon : HiOutlineCog;
  };

  const CurrentIcon = getWorkflowIcon(formData.workflow_type);

  return (
    <div className="min-h-screen" style={{
      background: 'linear-gradient(180deg, #0A1628 0%, #1A2942 50%, #0F1C2E 100%)'
    }}>
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Bouton retour */}
        <Link
          href="/workflows"
          className="inline-flex items-center gap-2 px-4 py-2 mb-6 rounded-lg font-medium text-sm transition-all duration-300"
          style={{
            background: 'rgba(30, 41, 59, 0.6)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(71, 85, 105, 0.5)',
            color: '#94A3B8'
          }}
        >
          <HiOutlineArrowLeft className="text-lg" />
          Retour à la liste des workflows
        </Link>

        {/* En-tête */}
        <motion.div 
          initial="hidden"
          animate="visible"
          variants={fadeInUp}
          className="mb-8 p-8 rounded-2xl" 
          style={{
            background: 'rgba(30, 41, 59, 0.4)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(71, 85, 105, 0.3)',
            boxShadow: '0 4px 24px rgba(0, 0, 0, 0.3)'
          }}
        >
          <h1 className="text-3xl font-bold mb-2" style={{ color: '#FFFFFF' }}>
            Créer un nouveau workflow
          </h1>
          <p style={{ color: '#94A3B8', fontSize: '15px' }}>
            Paramétrez votre workflow de calcul en quelques étapes
          </p>
        </motion.div>

        {/* Message d'erreur */}
        {error && (
          <div className="mb-6 p-4 rounded-xl" style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            backdropFilter: 'blur(12px)'
          }}>
            <div className="flex items-center gap-2">
              <svg className="h-5 w-5" style={{ color: '#EF4444' }} fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span style={{ color: '#FCA5A5' }}>{error}</span>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {/* Informations de base */}
          <motion.div 
            variants={fadeInUp}
            className="mb-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg" style={{
                background: 'rgba(59, 130, 246, 0.15)',
                border: '1px solid rgba(59, 130, 246, 0.3)'
              }}>
                <HiOutlineDocumentAdd className="text-xl" style={{ color: '#60A5FA' }} />
              </div>
              <h2 className="text-xl font-semibold" style={{ color: '#FFFFFF' }}>
                Informations de base
              </h2>
            </div>

            <div className="p-6 rounded-xl" style={{
              background: 'rgba(30, 41, 59, 0.4)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(71, 85, 105, 0.3)'
            }}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Nom */}
                <div>
                  <label htmlFor="name" className="block text-sm font-medium mb-2" style={{ color: '#CBD5E1' }}>
                    Nom du workflow <span style={{ color: '#60A5FA' }}>*</span>
                  </label>
                  <input
                    id="name"
                    name="name"
                    type="text"
                    required
                    value={formData.name}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-lg transition-all duration-300"
                    style={{
                      background: 'rgba(15, 23, 42, 0.5)',
                      border: '1px solid rgba(71, 85, 105, 0.4)',
                      color: '#FFFFFF',
                      outline: 'none'
                    }}
                    placeholder="Mon workflow de calcul"
                  />
                </div>

                {/* Type */}
                <div>
                  <label htmlFor="workflow_type" className="block text-sm font-medium mb-2" style={{ color: '#CBD5E1' }}>
                    Type de workflow <span style={{ color: '#60A5FA' }}>*</span>
                  </label>
                  <select
                    id="workflow_type"
                    name="workflow_type"
                    required
                    value={formData.workflow_type}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-lg transition-all duration-300 appearance-none"
                    style={{
                      background: 'rgba(15, 23, 42, 0.5)',
                      border: '1px solid rgba(71, 85, 105, 0.4)',
                      color: '#FFFFFF',
                      outline: 'none',
                      backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%2394A3B8' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e")`,
                      backgroundPosition: 'right 0.5rem center',
                      backgroundRepeat: 'no-repeat',
                      backgroundSize: '1.5em 1.5em',
                      paddingRight: '2.5rem'
                    }}
                  >
                    {workflowTypes.map(type => (
                      <option key={type.value} value={type.value} style={{ background: '#1A2942' }}>
                        {type.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Description */}
              <div className="mt-6">
                <label htmlFor="description" className="block text-sm font-medium mb-2" style={{ color: '#CBD5E1' }}>
                  Description
                </label>
                <textarea
                  id="description"
                  name="description"
                  rows={3}
                  value={formData.description}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-lg transition-all duration-300"
                  style={{
                    background: 'rgba(15, 23, 42, 0.5)',
                    border: '1px solid rgba(71, 85, 105, 0.4)',
                    color: '#FFFFFF',
                    outline: 'none',
                    resize: 'vertical'
                  }}
                  placeholder="Décrivez le but et les caractéristiques de votre workflow..."
                />
              </div>

              {/* Info type sélectionné */}
              <div className="mt-6 p-4 rounded-lg" style={{
                background: 'rgba(59, 130, 246, 0.1)',
                border: '1px solid rgba(59, 130, 246, 0.2)'
              }}>
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center" style={{
                    background: 'rgba(59, 130, 246, 0.2)',
                    border: '1px solid rgba(59, 130, 246, 0.3)'
                  }}>
                    <CurrentIcon className="text-lg" style={{ color: '#60A5FA' }} />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold mb-1" style={{ color: '#CBD5E1' }}>
                      {workflowTypes.find(wt => wt.value === formData.workflow_type)?.label}
                    </h3>
                    <p className="text-sm" style={{ color: '#94A3B8' }}>
                      {formData.workflow_type === 'ML_TRAINING' && "Ce type de workflow est optimisé pour les tâches d'apprentissage automatique."}
                      {formData.workflow_type === 'MATRIX_ADDITION' && "Ce type de workflow est optimisé pour les additions de matrices de grande taille."}
                      {formData.workflow_type === 'MATRIX_MULTIPLICATION' && "Ce type de workflow est optimisé pour les multiplications de matrices de grande taille."}
                      {formData.workflow_type === 'OPEN_MALARIA' && "Ce type de workflow est optimisé pour les simulations épidémiologiques complexes."}
                      {formData.workflow_type === 'CUSTOM' && "Vous avez sélectionné un type personnalisé. Vous pourrez configurer tous les paramètres manuellement."}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Paramètres d'exécution */}
          <motion.div 
            variants={fadeInUp}
            className="mb-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg" style={{
                background: 'rgba(59, 130, 246, 0.15)',
                border: '1px solid rgba(59, 130, 246, 0.3)'
              }}>
                <HiOutlineCog className="text-xl" style={{ color: '#60A5FA' }} />
              </div>
              <h2 className="text-xl font-semibold" style={{ color: '#FFFFFF' }}>
                Paramètres d'exécution
              </h2>
            </div>

            <div className="p-6 rounded-xl" style={{
              background: 'rgba(30, 41, 59, 0.4)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(71, 85, 105, 0.3)'
            }}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Executable path */}
                <div>
                  <label htmlFor="executable_path" className="block text-sm font-medium mb-2" style={{ color: '#CBD5E1' }}>
                    Chemin de l'exécutable
                  </label>
                  <div className="flex gap-2">
                    <input
                      id="executable_path"
                      name="executable_path"
                      type="text"
                      value={formData.executable_path}
                      onChange={handleChange}
                      className="flex-1 px-4 py-3 rounded-lg transition-all duration-300"
                      style={{
                        background: 'rgba(15, 23, 42, 0.5)',
                        border: '1px solid rgba(71, 85, 105, 0.4)',
                        color: '#FFFFFF',
                        outline: 'none'
                      }}
                      placeholder="/chemin/vers/executable.py"
                    />
                    <button
                      type="button"
                      onClick={() => handleSelectPath('executable_path')}
                      className="px-4 py-3 rounded-lg transition-all duration-300"
                      style={{
                        background: 'rgba(59, 130, 246, 0.2)',
                        border: '1px solid rgba(59, 130, 246, 0.3)',
                        color: '#60A5FA'
                      }}
                      title="Sélectionner un fichier"
                    >
                      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-5L9 5H5a2 2 0 00-2 2z" />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Input path */}
                <div>
                  <label htmlFor="input_path" className="block text-sm font-medium mb-2" style={{ color: '#CBD5E1' }}>
                    Chemin des données d'entrée
                  </label>
                  <div className="flex gap-2">
                    <input
                      id="input_path"
                      name="input_path"
                      type="text"
                      value={formData.input_path}
                      onChange={handleChange}
                      className="flex-1 px-4 py-3 rounded-lg transition-all duration-300"
                      style={{
                        background: 'rgba(15, 23, 42, 0.5)',
                        border: '1px solid rgba(71, 85, 105, 0.4)',
                        color: '#FFFFFF',
                        outline: 'none'
                      }}
                      placeholder="/chemin/vers/données/input/"
                    />
                    <button
                      type="button"
                      onClick={() => handleSelectPath('input_path')}
                      className="px-4 py-3 rounded-lg transition-all duration-300"
                      style={{
                        background: 'rgba(59, 130, 246, 0.2)',
                        border: '1px solid rgba(59, 130, 246, 0.3)',
                        color: '#60A5FA'
                      }}
                      title="Sélectionner un dossier"
                    >
                      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-5L9 5H5a2 2 0 00-2 2z" />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Output path */}
                <div className="md:col-span-2">
                  <label htmlFor="output_path" className="block text-sm font-medium mb-2" style={{ color: '#CBD5E1' }}>
                    Chemin des résultats
                  </label>
                  <div className="flex gap-2">
                    <input
                      id="output_path"
                      name="output_path"
                      type="text"
                      value={formData.output_path}
                      onChange={handleChange}
                      className="flex-1 px-4 py-3 rounded-lg transition-all duration-300"
                      style={{
                        background: 'rgba(15, 23, 42, 0.5)',
                        border: '1px solid rgba(71, 85, 105, 0.4)',
                        color: '#FFFFFF',
                        outline: 'none'
                      }}
                      placeholder="/chemin/vers/résultats/output/"
                    />
                    <button
                      type="button"
                      onClick={() => handleSelectPath('output_path')}
                      className="px-4 py-3 rounded-lg transition-all duration-300"
                      style={{
                        background: 'rgba(59, 130, 246, 0.2)',
                        border: '1px solid rgba(59, 130, 246, 0.3)',
                        color: '#60A5FA'
                      }}
                      title="Sélectionner un dossier"
                    >
                      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-5L9 5H5a2 2 0 00-2 2z" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Paramètres avancés */}
          <motion.div 
            variants={fadeInUp}
            className="mb-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg" style={{
                background: 'rgba(59, 130, 246, 0.15)',
                border: '1px solid rgba(59, 130, 246, 0.3)'
              }}>
                <HiOutlineClock className="text-xl" style={{ color: '#60A5FA' }} />
              </div>
              <h2 className="text-xl font-semibold" style={{ color: '#FFFFFF' }}>
                Paramètres avancés
              </h2>
            </div>

            <div className="p-6 rounded-xl" style={{
              background: 'rgba(30, 41, 59, 0.4)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(71, 85, 105, 0.3)'
            }}>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Priorité */}
                <div>
                  <label htmlFor="priority" className="block text-sm font-medium mb-2" style={{ color: '#CBD5E1' }}>
                    Priorité (1-10)
                  </label>
                  <input
                    id="priority"
                    name="priority"
                    type="number"
                    min="1"
                    max="10"
                    value={formData.priority}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-lg transition-all duration-300"
                    style={{
                      background: 'rgba(15, 23, 42, 0.5)',
                      border: '1px solid rgba(71, 85, 105, 0.4)',
                      color: '#FFFFFF',
                      outline: 'none'
                    }}
                  />
                  <div className="mt-1 text-xs" style={{ color: '#94A3B8' }}>
                    Plus la valeur est élevée, plus la priorité est haute
                  </div>
                </div>

                {/* Temps d'exécution */}
                <div>
                  <label htmlFor="max_execution_time" className="block text-sm font-medium mb-2" style={{ color: '#CBD5E1' }}>
                    Temps d'exécution max (s)
                  </label>
                  <input
                    id="max_execution_time"
                    name="max_execution_time"
                    type="number"
                    min="60"
                    value={formData.max_execution_time}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-lg transition-all duration-300"
                    style={{
                      background: 'rgba(15, 23, 42, 0.5)',
                      border: '1px solid rgba(71, 85, 105, 0.4)',
                      color: '#FFFFFF',
                      outline: 'none'
                    }}
                  />
                  <div className="mt-1 text-xs" style={{ color: '#94A3B8' }}>
                    Durée maximale en secondes avant arrêt forcé
                  </div>
                </div>

                {/* Retry count */}
                <div>
                  <label htmlFor="retry_count" className="block text-sm font-medium mb-2" style={{ color: '#CBD5E1' }}>
                    Nombre de tentatives
                  </label>
                  <input
                    id="retry_count"
                    name="retry_count"
                    type="number"
                    min="0"
                    max="10"
                    value={formData.retry_count.toString()}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-lg transition-all duration-300"
                    style={{
                      background: 'rgba(15, 23, 42, 0.5)',
                      border: '1px solid rgba(71, 85, 105, 0.4)',
                      color: '#FFFFFF',
                      outline: 'none'
                    }}
                  />
                  <div className="mt-1 text-xs" style={{ color: '#94A3B8' }}>
                    Nombre de réessais en cas d'échec
                  </div>
                </div>
              </div>

              {/* Conseils */}
              <div className="mt-6 p-4 rounded-lg" style={{
                background: 'rgba(59, 130, 246, 0.1)',
                border: '1px solid rgba(59, 130, 246, 0.2)'
              }}>
                <div className="flex items-start gap-3">
                  <svg className="h-5 w-5 flex-shrink-0 mt-0.5" style={{ color: '#60A5FA' }} viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  <div>
                    <h4 className="text-sm font-medium mb-1" style={{ color: '#CBD5E1' }}>
                      Conseils pour les paramètres avancés
                    </h4>
                    <ul className="text-sm space-y-1" style={{ color: '#94A3B8' }}>
                      <li>• Une priorité élevée (8-10) est recommandée pour les tâches urgentes</li>
                      <li>• Le temps d'exécution doit être suffisant pour permettre l'achèvement du workflow</li>
                      <li>• Définissez plusieurs tentatives pour les workflows avec des dépendances externes</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Boutons d'action */}
          <motion.div 
            variants={fadeInUp}
            className="flex justify-end gap-4 pt-4"
          >
            <Link
              href="/workflows"
              className="py-3 px-6 rounded-lg font-medium transition-all duration-300"
              style={{
                background: 'rgba(30, 41, 59, 0.6)',
                backdropFilter: 'blur(12px)',
                border: '1px solid rgba(71, 85, 105, 0.5)',
                color: '#94A3B8'
              }}
            >
              Annuler
            </Link>
            <button
              type="submit"
              disabled={loading}
              className="py-3 px-8 rounded-lg font-medium transition-all duration-300"
              style={{
                background: loading ? 'rgba(59, 130, 246, 0.3)' : 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
                border: '1px solid rgba(59, 130, 246, 0.4)',
                color: '#FFFFFF',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.7 : 1
              }}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Création en cours...
                </span>
              ) : (
                'Créer le workflow'
              )}
            </button>
          </motion.div>
        </form>
      </div>
    </div>
  );
}