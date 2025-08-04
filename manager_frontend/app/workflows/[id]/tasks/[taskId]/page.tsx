"use client"

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { taskService, volunteerService } from '@/lib/api';
import { Task, Volunteer } from '../../../../../lib/types';
import Link from 'next/link';

export default function TaskDetailPage() {
  const router = useRouter();
  const params = useParams();
  const workflowId = params.id as string;
  const taskId = params.taskId as string;
  
  const [task, setTask] = useState<Task | null>(null);
  const [volunteers, setVolunteers] = useState<Volunteer[]>([]);
  const [availableVolunteers, setAvailableVolunteers] = useState<Volunteer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAssignForm, setShowAssignForm] = useState(false);
  const [selectedVolunteerId, setSelectedVolunteerId] = useState<string>('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Récupérer les détails de la tâche
        const taskData = await taskService.getTask(taskId);
        setTask(taskData);
        
        // Récupérer les volontaires assignés à cette tâche
        const taskVolunteers = await taskService.getTaskVolunteers(taskId);
        setVolunteers(taskVolunteers);
        
        // Récupérer tous les volontaires disponibles
        const allVolunteers = await volunteerService.getVolunteers();
        const available: Volunteer[] = allVolunteers.filter(
          (v: Volunteer) => v.available && !taskVolunteers.some((tv: Volunteer) => tv.id === v.id)
        );
        setAvailableVolunteers(available);
        
        setLoading(false);
      } catch (err: any) {
        setError(err.error || 'Une erreur est survenue');
        setLoading(false);
      }
    };

    fetchData();
  }, [taskId]);

  const handleAssignVolunteer = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedVolunteerId) return;
    
    try {
      setLoading(true);
      await taskService.assignTask(taskId, selectedVolunteerId);
      
      // Actualiser les données
      const taskVolunteers = await taskService.getTaskVolunteers(taskId);
      setVolunteers(taskVolunteers);
      
      const allVolunteers = await volunteerService.getVolunteers();
      const available: Volunteer[] = allVolunteers.filter(
        (v: Volunteer) => v.available && !taskVolunteers.some((tv: Volunteer) => tv.id === v.id)
      );
      setAvailableVolunteers(available);
      
      setShowAssignForm(false);
      setSelectedVolunteerId('');
      setLoading(false);
    } catch (err: any) {
      setError(err.error || 'Une erreur est survenue lors de l\'assignation du volontaire');
      setLoading(false);
    }
  };

  const getStatusClass = (status: string) => {
    switch (status) {
      case 'PENDING':
        return {
          bg: 'bg-gradient-to-r from-amber-50 to-orange-50 border-amber-200',
          text: 'text-amber-800',
          icon: '⏳',
          dot: 'bg-amber-400'
        };
      case 'RUNNING':
        return {
          bg: 'bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200',
          text: 'text-blue-800',
          icon: '▶️',
          dot: 'bg-blue-400'
        };
      case 'COMPLETED':
        return {
          bg: 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-200',
          text: 'text-green-800',
          icon: '✅',
          dot: 'bg-green-400'
        };
      case 'FAILED':
        return {
          bg: 'bg-gradient-to-r from-red-50 to-rose-50 border-red-200',
          text: 'text-red-800',
          icon: '❌',
          dot: 'bg-red-400'
        };
      case 'ASSIGNED':
        return {
          bg: 'bg-gradient-to-r from-purple-50 to-violet-50 border-purple-200',
          text: 'text-purple-800',
          icon: '🔗',
          dot: 'bg-purple-400'
        };
      default:
        return {
          bg: 'bg-gradient-to-r from-gray-50 to-slate-50 border-gray-200',
          text: 'text-gray-800',
          icon: '❓',
          dot: 'bg-gray-400'
        };
    }
  };

  if (loading && !task) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-black">
        <div className="container mx-auto p-6">
          <div className="flex justify-center items-center h-64">
            <div className="relative">
              <div className="animate-spin rounded-full h-16 w-16 border-4 border-slate-600"></div>
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-blue-500 absolute top-0"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-black">
        <div className="container mx-auto p-6">
          <div className="bg-gradient-to-r from-red-50 to-rose-50 border-l-4 border-red-400 text-red-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
            <div className="flex items-center">
              <div className="flex-shrink-0 w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                <svg className="h-6 w-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <p className="ml-4 font-medium text-lg">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Format JSON pour une meilleure présentation
  const formatJSON = (json: any) => {
    if (!json) return 'Non défini';
    try {
      if (typeof json === 'string') {
        return JSON.parse(json);
      }
      // Si c'est un objet simple, afficher ses propriétés directement
      if (typeof json === 'object' && Object.keys(json).length < 4) {
        return Object.entries(json)
          .map(([key, value]) => `${key}: ${value}`)
          .join(', ');
      }
      // Sinon formater en JSON
      return JSON.stringify(json, null, 2);
    } catch (e) {
      return json.toString();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-black">
      <div className="container mx-auto p-6 max-w-7xl">
        {/* Navigation */}
        <div className="mb-8">
          <Link  href={`/workflows/${workflowId}`} 
            className="group inline-flex items-center px-6 py-3 rounded-2xl bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5">
            <svg className="h-5 w-5 mr-3 group-hover:-translate-x-1 transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Retour aux tâches
          </Link>
        </div>

        {task && (
          <div className="space-y-8">
            {/* En-tête de la tâche */}
            <div className="bg-white/95 backdrop-blur-sm rounded-3xl shadow-2xl overflow-hidden border border-white/20">
              <div className="bg-gradient-to-r from-indigo-600 via-blue-600 to-purple-700 px-8 py-6 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20"></div>
                <div className="relative flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
                  <div>
                    <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">{task.name}</h1>
                    <p className="text-blue-100 opacity-90">ID: {task.id}</p>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className={`px-5 py-3 rounded-2xl border backdrop-blur-sm ${getStatusClass(task.status).bg} ${getStatusClass(task.status).text} flex items-center shadow-lg`}>
                      <div className={`w-2 h-2 rounded-full ${getStatusClass(task.status).dot} mr-3 animate-pulse`}></div>
                      <span className="mr-2 text-lg">{getStatusClass(task.status).icon}</span>
                      <span className="font-semibold">{task.status}</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="p-8">
                <div className="mb-8">
                  <div className="bg-gradient-to-r from-slate-50 to-blue-50 p-6 rounded-2xl border-l-4 border-blue-500 shadow-inner">
                    <p className="text-gray-800 text-lg leading-relaxed">{task.description}</p>
                  </div>
                </div>
                
                {/* Grille des détails */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-10">
                  {/* Détails de la tâche */}
                  <div className="bg-gradient-to-br from-gray-50 to-slate-50 p-6 rounded-2xl shadow-lg border border-gray-100">
                    <div className="flex items-center mb-6">
                      <div className="w-10 h-10 bg-blue-100 rounded-xl flex items-center justify-center mr-4">
                        <svg className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-bold text-gray-800">Détails de la tâche</h3>
                    </div>
                    <div className="space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center py-3 border-b border-gray-200 last:border-b-0">
                        <span className="font-semibold text-gray-700 w-40 mb-1 sm:mb-0">Workflow:</span>
                        <span className="text-gray-900 bg-white px-3 py-1 rounded-lg shadow-sm">{task.workflow_name}</span>
                      </div>
                      <div className="flex flex-col sm:flex-row sm:items-center py-3 border-b border-gray-200 last:border-b-0">
                        <span className="font-semibold text-gray-700 w-40 mb-1 sm:mb-0">Commande:</span>
                        <code className="bg-gray-800 text-green-400 px-4 py-2 rounded-lg font-mono text-sm shadow-inner">{task.command}</code>
                      </div>
                      <div className="py-3 border-b border-gray-200 last:border-b-0">
                        <span className="font-semibold text-gray-700 block mb-2">Paramètres:</span>
                        <pre className="bg-gray-800 text-gray-100 p-4 rounded-xl text-sm overflow-x-auto font-mono shadow-inner border">{formatJSON(task.parameters)}</pre>
                      </div>
                      <div className="py-3 border-b border-gray-200 last:border-b-0">
                        <span className="font-semibold text-gray-700 block mb-2">Ressources requises:</span>
                        <pre className="bg-gray-800 text-gray-100 p-4 rounded-xl text-sm overflow-x-auto font-mono shadow-inner border">{formatJSON(task.required_resources)}</pre>
                      </div>
                      <div className="flex flex-col sm:flex-row sm:items-center py-3 border-b border-gray-200 last:border-b-0">
                        <span className="font-semibold text-gray-700 w-40 mb-1 sm:mb-0">Temps max estimé:</span>
                        <span className="text-gray-900 bg-amber-50 px-3 py-1 rounded-lg shadow-sm border border-amber-100">{task.estimated_max_time} secondes</span>
                      </div>
                      <div className="flex flex-col sm:flex-row sm:items-center py-3 border-b border-gray-200 last:border-b-0">
                        <span className="font-semibold text-gray-700 w-40 mb-1 sm:mb-0">Créé le:</span>
                        <span className="text-gray-900 bg-white px-3 py-1 rounded-lg shadow-sm">{new Date(task.created_at).toLocaleString('fr-FR')}</span>
                      </div>
                      {task.start_time && (
                        <div className="flex flex-col sm:flex-row sm:items-center py-3 border-b border-gray-200 last:border-b-0">
                          <span className="font-semibold text-gray-700 w-40 mb-1 sm:mb-0">Démarré le:</span>
                          <span className="text-gray-900 bg-green-50 px-3 py-1 rounded-lg shadow-sm border border-green-100">{new Date(task.start_time).toLocaleString('fr-FR')}</span>
                        </div>
                      )}
                      {task.end_time && (
                        <div className="flex flex-col sm:flex-row sm:items-center py-3 border-b border-gray-200 last:border-b-0">
                          <span className="font-semibold text-gray-700 w-40 mb-1 sm:mb-0">Terminé le:</span>
                          <span className="text-gray-900 bg-blue-50 px-3 py-1 rounded-lg shadow-sm border border-blue-100">{new Date(task.end_time).toLocaleString('fr-FR')}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Progression */}
                  <div className="bg-gradient-to-br from-slate-50 to-gray-50 p-6 rounded-2xl shadow-lg border border-gray-100">
                    <div className="flex items-center mb-6">
                      <div className="w-10 h-10 bg-green-100 rounded-xl flex items-center justify-center mr-4">
                        <svg className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-bold text-gray-800">Progression</h3>
                    </div>
                    
                    <div className="mb-8">
                      <div className="flex justify-between mb-3">
                        <span className="text-sm font-semibold text-gray-700">Avancement</span>
                        <span className="text-sm font-bold text-gray-900 bg-white px-2 py-1 rounded-lg shadow-sm">{task.progress}%</span>
                      </div>
                      <div className="w-full h-6 bg-gray-200 rounded-full overflow-hidden shadow-inner">
                        <div 
                          className={`h-full rounded-full transition-all duration-700 ease-out ${
                            task.status === 'FAILED' ? 'bg-gradient-to-r from-red-500 to-red-600' : 
                            task.status === 'COMPLETED' ? 'bg-gradient-to-r from-green-500 to-green-600' : 
                            'bg-gradient-to-r from-blue-500 to-blue-600'
                          } shadow-lg`} 
                          style={{ width: `${task.progress}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    {task.start_time && task.end_time ? (
                      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 rounded-2xl border border-blue-100 shadow-inner">
                        <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
                          <svg className="h-5 w-5 text-blue-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Durée d'exécution
                        </h4>
                        <div className="text-2xl font-bold text-blue-800">
                          {((new Date(task.end_time).getTime() - new Date(task.start_time).getTime()) / 1000).toFixed(2)} secondes
                        </div>
                      </div>
                    ) : task.start_time ? (
                      <div className="bg-gradient-to-r from-orange-50 to-amber-50 p-5 rounded-2xl border border-orange-100 shadow-inner">
                        <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
                          <svg className="h-5 w-5 text-orange-500 mr-2 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          En cours depuis
                        </h4>
                        <div className="text-2xl font-bold text-orange-800">
                          {((new Date().getTime() - new Date(task.start_time).getTime()) / 1000).toFixed(2)} secondes
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
                
                {/* Section des volontaires */}
                <div className="mb-10">
                  <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
                    <div className="flex items-center">
                      <div className="w-10 h-10 bg-purple-100 rounded-xl flex items-center justify-center mr-4">
                        <svg className="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-bold text-gray-800">
                        Volontaires assignés 
                        <span className="ml-3 px-3 py-1 bg-gradient-to-r from-purple-100 to-purple-200 text-purple-800 rounded-full text-sm font-semibold shadow-sm">
                          {volunteers.length}
                        </span>
                      </h3>
                    </div>
                    <button
                      onClick={() => setShowAssignForm(!showAssignForm)}
                      className="group inline-flex items-center px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-semibold rounded-2xl transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
                    >
                      <svg className="h-5 w-5 mr-2 group-hover:rotate-90 transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                      </svg>
                      Assigner un volontaire
                    </button>
                  </div>
                  
                  {showAssignForm && (
                    <div className="bg-gradient-to-br from-gray-50 to-slate-50 p-6 rounded-2xl shadow-lg mb-8 border border-gray-200">
                      <h4 className="text-lg font-bold mb-4 text-gray-800 flex items-center">
                        <svg className="h-5 w-5 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                        </svg>
                        Assigner un nouveau volontaire
                      </h4>
                      {availableVolunteers.length === 0 ? (
                        <div className="bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 text-amber-800 p-5 rounded-2xl shadow-inner">
                          <div className="flex items-center">
                            <svg className="h-6 w-6 text-amber-500 mr-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                            <p className="font-medium">Aucun volontaire disponible pour cette tâche. Les volontaires doivent être disponibles et non déjà assignés.</p>
                          </div>
                        </div>
                      ) : (
                        <form onSubmit={handleAssignVolunteer} className="flex flex-col lg:flex-row lg:space-x-4 gap-4">
                          <div className="relative flex-grow">
                            <select
                              className="block w-full bg-white border-2 border-gray-200 hover:border-gray-300 px-4 py-4 pr-10 rounded-2xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                              value={selectedVolunteerId}
                              onChange={(e) => setSelectedVolunteerId(e.target.value)}
                              required
                            >
                              <option value="">Sélectionnez un volontaire</option>
                              {availableVolunteers.map((volunteer) => (
                                <option key={volunteer.id} value={volunteer.id}>
                                  {volunteer.name} ({volunteer.hostname}) - {volunteer.cpu_cores} cœurs, {volunteer.ram_mb} MB RAM
                                </option>
                              ))}
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500">
                              <svg className="fill-current h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                              </svg>
                            </div>
                          </div>
                          <div className="flex space-x-3">
                            <button
                              type="submit"
                              className="group flex-grow lg:flex-grow-0 inline-flex justify-center items-center px-6 py-4 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold rounded-2xl focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
                            >
                              <svg className="h-5 w-5 mr-2 group-hover:scale-110 transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                              </svg>
                              Assigner
                            </button>
                            <button
                              type="button"
                              onClick={() => setShowAssignForm(false)}
                              className="group flex-grow lg:flex-grow-0 inline-flex justify-center items-center px-6 py-4 bg-gradient-to-r from-gray-200 to-gray-300 hover:from-gray-300 hover:to-gray-400 text-gray-800 font-semibold rounded-2xl focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
                            >
                              <svg className="h-5 w-5 mr-2 group-hover:rotate-45 transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                              </svg>
                              Annuler
                            </button>
                          </div>
                        </form>
                      )}
                    </div>
                  )}
                </div>

                {/* Tableau des volontaires assignés */}
                {volunteers.length > 0 && (
                  <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-2xl p-6 border border-gray-200">
                    <h3 className="text-xl font-bold text-gray-800 mb-4">Détails des volontaires</h3>
                    <table className="min-w-full divide-y divide-gray-200 rounded-lg overflow-hidden">
                      <thead className="bg-gradient-to-r from-blue-50 to-indigo-100">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">Nom</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">Ressources</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">Statut</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">Dernière activité</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-100">
                        {volunteers.map((v) => {
                          const status = getStatusClass(v.status);
                          return (
                            <tr key={v.id} className="hover:bg-gray-50">
                              <td className="px-6 py-4 text-sm text-gray-900 font-medium">{v.name}</td>
                              <td className="px-6 py-4 text-sm text-gray-600">
                                {v.cpu_cores} cœurs, {v.ram_mb} MB RAM, {v.disk_gb} GB
                              </td>
                              <td className="px-6 py-4">
                                <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${status.bg} ${status.text}`}>
                                  {status.icon} {v.status}
                                </span>
                              </td>
                              <td className="px-6 py-4 text-sm text-gray-500">
                                {new Date(v.last_seen).toLocaleString('fr-FR')}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Sous-tâches */}
                {task.subtasks && task.subtasks.length > 0 && (
                  <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-2xl p-6 border border-gray-200">
                    <h3 className="text-xl font-bold text-gray-800 mb-6">Sous-tâches</h3>
                    <ul className="space-y-4">
                      {task.subtasks.map((subtask) => {
                        const status = getStatusClass(subtask.status);
                        return (
                          <li key={subtask.id} className="p-4 bg-gradient-to-r from-gray-50 to-slate-100 rounded-xl shadow-sm border border-gray-100">
                            <div className="flex justify-between items-center">
                              <div>
                                <h4 className="font-semibold text-gray-900 text-lg">{subtask.name}</h4>
                                {subtask.description && (
                                  <p className="text-gray-600 text-sm mt-1">{subtask.description}</p>
                                )}
                              </div>
                              <div className={`px-3 py-1 rounded-full text-xs font-semibold ${status.bg} ${status.text} flex items-center`}>
                                <span className="mr-2">{status.icon}</span> {subtask.status}
                              </div>
                            </div>
                            <div className="mt-3">
                              <div className="w-full bg-gray-200 rounded-full h-2.5">
                                <div
                                  className={`h-full rounded-full ${
                                    subtask.status === 'FAILED'
                                      ? 'bg-red-500'
                                      : subtask.status === 'COMPLETED'
                                      ? 'bg-green-500'
                                      : 'bg-blue-500'
                                  }`}
                                  style={{ width: `${subtask.progress}%` }}
                                ></div>
                              </div>
                              <p className="text-sm text-gray-500 mt-1">{subtask.progress}%</p>
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                )}

                {/* Logs */}
                {task.logs && (
                  <div className="bg-gradient-to-br from-black via-gray-800 to-slate-900 text-white p-6 mt-10 rounded-2xl shadow-2xl border border-gray-700">
                    <h3 className="text-xl font-bold mb-4">Logs d'exécution</h3>
                    <pre className="bg-black/60 rounded-lg p-4 text-sm font-mono overflow-x-auto whitespace-pre-wrap">{task.logs}</pre>
                  </div>
              )}

            </div>
          </div>
        </div>
        )}
      </div>
    </div>
    
);
}



                              