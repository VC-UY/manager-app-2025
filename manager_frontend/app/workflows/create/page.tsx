'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { workflowService } from '@/lib/api';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { HiOutlineDocumentAdd, HiOutlineCog, HiOutlineClock, HiOutlineArrowLeft } from 'react-icons/hi';

// Types des workflows avec des icônes associées
const workflowTypes = [
  { value: 'MATRIX_ADDITION', label: 'Addition de matrices de grande taille', icon: '➕' },
  { value: 'MATRIX_MULTIPLICATION', label: 'Multiplication de matrices de grande taille', icon: '✖️' },
  { value: 'ML_TRAINING', label: 'Entraînement de modèle machine learning', icon: '🧠' },
  { value: 'OPEN_MALARIA', label: 'Simulation de propagation de la malaria', icon: '🦟' },
  { value: 'CUSTOM', label: 'Workflow personnalisé', icon: '🔧' }
];

export default function CreateWorkflowPage() {
  const router = useRouter();

  // État du formulaire avec des valeurs par défaut qui sont des strings pour éviter NaN
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

  // État de soumission
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Animation pour les sections du formulaire
  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };


  
  // Gestion des changements de champs avec sécurité contre NaN
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    
    // Pour les champs numériques, vérifier et convertir proprement les valeurs
    if (name === 'priority' || name === 'max_execution_time' || name === 'retry_count') {
      const numValue = parseInt(value, 10);
      setFormData(prev => ({
        ...prev,
        // Utiliser la valeur numérique seulement si c'est un nombre valide
        [name]: isNaN(numValue) ? 0 : numValue
      }));
    } else {
      // Pour les valeurs textuelles
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  // Fonction pour gérer la sélection de chemin
// Version simplifiée sans déclarations de types globales
const handleSelectPath = async (fieldName: string): Promise<void> => {
  try {
    // Vérifier si l'API est disponible
    const hasFileSystemAccess = 'showOpenFilePicker' in window || 'showDirectoryPicker' in window;
    
    if (hasFileSystemAccess) {
      if (fieldName === 'executable_path') {
        // Sélection de fichier
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
        // Sélection de dossier
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
    
    // Fallback vers input file classique
    useFallbackFilePicker(fieldName);
    
  } catch (error: unknown) {
    const err = error as { name?: string };
    if (err.name !== 'AbortError') {
      console.error('Erreur lors de la sélection du chemin:', error);
      useFallbackFilePicker(fieldName);
    }
  }
};

// Fonction fallback
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

  // Soumission du formulaire
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Vérifier les valeurs numériques avant l'envoi
      const dataToSubmit = {
        ...formData,
        // S'assurer que les valeurs numériques sont des nombres valides
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

  // Fonction pour obtenir l'icône en fonction du type de workflow
  interface WorkflowType {
    value: string;
    label: string;
    icon: string;
  }

  const getWorkflowIcon = (type: string): string => {
    const workflowType: WorkflowType | undefined = workflowTypes.find(wt => wt.value === type);
    return workflowType ? workflowType.icon : '🔧';
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4 max-w-4xl">
        <Link
          href="/workflows"
          className="inline-flex items-center text-indigo-700 hover:text-indigo-900 font-medium mb-6 transition-colors"
        >
          <HiOutlineArrowLeft className="mr-2" /> Retour à la liste des workflows
        </Link>
        
        <motion.div 
          initial="hidden"
          animate="visible"
          variants={fadeInUp}
          className="bg-white rounded-xl shadow-xl overflow-hidden border border-indigo-100"
        >
          <div className="bg-gradient-to-r from-indigo-700 to-blue-600 px-8 py-7 text-white">
            <h1 className="text-3xl font-bold drop-shadow-sm">Créer un nouveau workflow</h1>
            <p className="mt-2 opacity-90 text-indigo-50">Paramétrez votre workflow de calcul en quelques étapes</p>
          </div>

          {error && (
            <div className="mx-8 mt-6 bg-red-50 border-l-4 border-red-500 text-red-700 p-4 rounded shadow-sm">
              <div className="flex items-center">
                <svg className="h-5 w-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                {error}
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="p-8">
            {/* Informations de base */}
            <motion.div 
              variants={fadeInUp}
              className="mb-8"
            >
              <div className="flex items-center mb-4">
                <div className="bg-indigo-100 p-2 rounded-lg mr-3">
                  <HiOutlineDocumentAdd className="text-2xl text-indigo-700" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Informations de base</h2>
              </div>
              
              <div className="p-6 bg-white rounded-lg border border-indigo-200 shadow-sm">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-900 mb-1">
                      Nom du workflow <span className="text-indigo-600">*</span>
                    </label>
                    <input
                      id="name"
                      name="name"
                      type="text"
                      required
                      value={formData.name}
                      onChange={handleChange}
                      className="block w-full border border-gray-300 rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                      placeholder="Mon workflow de calcul"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="workflow_type" className="block text-sm font-medium text-gray-900 mb-1">
                      Type de workflow <span className="text-indigo-600">*</span>
                    </label>
                    <div className="relative">
                      <select
                        id="workflow_type"
                        name="workflow_type"
                        required
                        value={formData.workflow_type}
                        onChange={handleChange}
                        className="block w-full appearance-none border border-gray-300 rounded-lg shadow-sm py-3 px-4 pr-10 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all bg-white"
                      >
                        {workflowTypes.map(type => (
                          <option key={type.value} value={type.value}>
                            {type.icon} {type.label}
                          </option>
                        ))}
                      </select>
                      <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                        <svg className="h-5 w-5 text-gray-500" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-4">
                  <label htmlFor="description" className="block text-sm font-medium text-gray-900 mb-1">
                    Description
                  </label>
                  <textarea
                    id="description"
                    name="description"
                    rows={3}
                    value={formData.description}
                    onChange={handleChange}
                    className="block w-full border border-gray-300 rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                    placeholder="Décrivez le but et les caractéristiques de votre workflow..."
                  />
                </div>
                
                <div className="mt-5 p-4 bg-indigo-50 border border-indigo-100 rounded-lg shadow-inner">
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      <div className="flex items-center justify-center h-12 w-12 rounded-full bg-indigo-600 text-white shadow-md">
                        {getWorkflowIcon(formData.workflow_type)}
                      </div>
                    </div>
                    <div className="ml-4">
                      <h3 className="text-sm font-semibold text-indigo-900">Type sélectionné: {workflowTypes.find(wt => wt.value === formData.workflow_type)?.label}</h3>
                      <p className="mt-1 text-sm text-indigo-700">
                        {formData.workflow_type === 'ML_TRAINING' && "Ce type de workflow est optimisé pour les tâches d'apprentissage automatique."}
                        {formData.workflow_type === 'MATRIX_ADDITION' && "Ce type de workflow est optimisé pour les additions de matrices de grande taille."}
                        {formData.workflow_type === 'MATRIX_MULTIPLICATION' && "Ce type de workflow est optimisé pour les multiplications de matrices de grande taille."}
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
              className="mb-8"
            >
              <div className="flex items-center mb-4">
                <div className="bg-indigo-100 p-2 rounded-lg mr-3">
                  <HiOutlineCog className="text-2xl text-indigo-700" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Paramètres d'exécution</h2>
              </div>
              
              <div className="p-6 bg-white rounded-lg border border-indigo-200 shadow-sm">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="executable_path" className="block text-sm font-medium text-gray-900 mb-1">
                      Chemin de l'exécutable
                    </label>
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                          <svg className="h-5 w-5 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                          </svg>
                        </div>
                        <input
                          id="executable_path"
                          name="executable_path"
                          type="text"
                          value={formData.executable_path}
                          onChange={handleChange}
                          className="block w-full pl-10 border border-gray-300 rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                          placeholder="/chemin/vers/executable.py"
                        />
                      </div>
                      <button
                        type="button"
                        onClick={() => handleSelectPath('executable_path')}
                        className="px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all flex items-center"
                        title="Sélectionner un fichier"
                      >
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-5L9 5H5a2 2 0 00-2 2z" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  
                  <div>
                    <label htmlFor="input_path" className="block text-sm font-medium text-gray-900 mb-1">
                      Chemin des données d'entrée
                    </label>
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                          <svg className="h-5 w-5 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                          </svg>
                        </div>
                        <input
                          id="input_path"
                          name="input_path"
                          type="text"
                          value={formData.input_path}
                          onChange={handleChange}
                          className="block w-full pl-10 border border-gray-300 rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                          placeholder="/chemin/vers/données/input/"
                        />
                      </div>
                      <button
                        type="button"
                        onClick={() => handleSelectPath('input_path')}
                        className="px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all flex items-center"
                        title="Sélectionner un dossier"
                      >
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-5L9 5H5a2 2 0 00-2 2z" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  
                  <div className="md:col-span-2">
                    <label htmlFor="output_path" className="block text-sm font-medium text-gray-900 mb-1">
                      Chemin des résultats
                    </label>
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                          <svg className="h-5 w-5 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                          </svg>
                        </div>
                        <input
                          id="output_path"
                          name="output_path"
                          type="text"
                          value={formData.output_path}
                          onChange={handleChange}
                          className="block w-full pl-10 border border-gray-300 rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                          placeholder="/chemin/vers/résultats/output/"
                        />
                      </div>
                      <button
                        type="button"
                        onClick={() => handleSelectPath('output_path')}
                        className="px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all flex items-center"
                        title="Sélectionner un dossier"
                      >
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-5L9 5H5a2 2 0 00-2 2z" />
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
              className="mb-8"
            >
              <div className="flex items-center mb-4">
                <div className="bg-indigo-100 p-2 rounded-lg mr-3">
                  <HiOutlineClock className="text-2xl text-indigo-700" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Paramètres avancés</h2>
              </div>
              
              <div className="p-6 bg-white rounded-lg border border-indigo-200 shadow-sm">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <label htmlFor="priority" className="block text-sm font-medium text-gray-900 mb-1">
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
                        className="block w-full border border-gray-300 rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                      />
                      <div className="mt-1 text-sm text-indigo-800 font-medium">Plus la valeur est élevée, plus la priorité est haute</div>
                    </div>
                  </div>
                  
                  <div>
                    <label htmlFor="max_execution_time" className="block text-sm font-medium text-gray-900 mb-1">
                      Temps d'exécution max (s)
                    </label>
                    <div className="relative">
                      <input
                        id="max_execution_time"
                        name="max_execution_time"
                        type="number"
                        min="60"
                        value={formData.max_execution_time}
                        onChange={handleChange}
                        className="block w-full border border-gray-300 rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                      />
                      <div className="mt-1 text-sm text-indigo-800 font-medium">Durée maximale en secondes avant arrêt forcé</div>
                    </div>
                  </div>
                  
                  <div>
                    <label htmlFor="retry_count" className="block text-sm font-medium text-gray-900 mb-1">
                      Nombre de tentatives
                    </label>
                    <div className="relative">
                      <input
                        id="retry_count"
                        name="retry_count"
                        type="number"
                        min="0"
                        max="10"
                        value={formData.retry_count.toString()}
                        onChange={handleChange}
                        className="block w-full border border-gray-300 rounded-lg shadow-sm py-3 px-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                      />
                      <div className="mt-1 text-sm text-indigo-800 font-medium">Nombre de réessais en cas d'échec (0 = aucun)</div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-4 p-4 bg-blue-50 border border-blue-100 rounded-lg">
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <h4 className="text-sm font-medium text-blue-800">Conseils pour les paramètres avancés</h4>
                      <ul className="mt-1 text-sm text-blue-700 list-disc list-inside">
                        <li>Une priorité élevée (8-10) est recommandée pour les tâches urgentes</li>
                        <li>Le temps d'exécution doit être suffisant pour permettre l'achèvement du workflow</li>
                        <li>Définissez plusieurs tentatives pour les workflows avec des dépendances externes</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Boutons d'action */}
            <motion.div 
              variants={fadeInUp}
              className="flex justify-end space-x-4 pt-4"
            >
              <Link
                href="/workflows"
                className="py-3 px-6 rounded-lg text-gray-700 bg-gray-200 hover:bg-gray-300 font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 shadow-sm"
              >
                Annuler
              </Link>
              <button
                type="submit"
                disabled={loading}
                className={`py-3 px-8 rounded-lg font-medium text-white bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all ${
                  loading ? 'opacity-70 cursor-not-allowed' : ''
                }`}
              >
                {loading ? (
                  <span className="flex items-center">
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
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
        </motion.div>
      </div>
    </div>
  );
}