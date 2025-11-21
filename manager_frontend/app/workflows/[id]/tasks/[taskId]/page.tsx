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
          bg: 'bg-gradient-to-r from-amber-500/10 to-orange-500/10 border-amber-400/30',
          text: 'text-amber-400',
          icon: '⏳',
          dot: 'bg-amber-400'
        };
      case 'RUNNING':
        return {
          bg: 'bg-gradient-to-r from-blue-500/10 to-indigo-500/10 border-blue-400/30',
          text: 'text-blue-400',
          icon: '▶️',
          dot: 'bg-blue-400'
        };
      case 'COMPLETED':
        return {
          bg: 'bg-gradient-to-r from-green-500/10 to-emerald-500/10 border-green-400/30',
          text: 'text-green-400',
          icon: '✅',
          dot: 'bg-green-400'
        };
      case 'FAILED':
        return {
          bg: 'bg-gradient-to-r from-red-500/10 to-rose-500/10 border-red-400/30',
          text: 'text-red-400',
          icon: '❌',
          dot: 'bg-red-400'
        };
      case 'ASSIGNED':
        return {
          bg: 'bg-gradient-to-r from-purple-500/10 to-violet-500/10 border-purple-400/30',
          text: 'text-purple-400',
          icon: '🔗',
          dot: 'bg-purple-400'
        };
      default:
        return {
          bg: 'bg-gradient-to-r from-gray-500/10 to-slate-500/10 border-gray-400/30',
          text: 'text-gray-400',
          icon: '❓',
          dot: 'bg-gray-400'
        };
    }
  };

  if (loading && !task) {
    return (
      <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #001440 0%, #002060 50%, #001440 100%)' }}>
        <div className="container mx-auto p-6">
          <div className="flex justify-center items-center h-64">
            <div className="relative">
              <div className="animate-spin rounded-full h-16 w-16 border-4 border-slate-600"></div>
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-blue-500 absolute top-0" style={{ borderTopColor: '#00D4FF' }}></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #001440 0%, #002060 50%, #001440 100%)' }}>
        <div className="container mx-auto p-6">
          <div className="backdrop-blur-xl rounded-2xl p-6 border"
            style={{
              background: 'linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, rgba(185, 28, 28, 0.05) 100%)',
              borderColor: 'rgba(248, 113, 113, 0.3)',
              boxShadow: '0 8px 32px rgba(220, 38, 38, 0.2)',
            }}>
            <div className="flex items-center">
              <div className="flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(185, 28, 28, 0.1) 100%)',
                  border: '2px solid rgba(248, 113, 113, 0.3)',
                }}>
                <svg className="h-6 w-6" style={{ color: '#F87171' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <p className="ml-4 font-medium text-lg" style={{ color: '#F87171' }}>{error}</p>
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
      if (typeof json === 'object' && Object.keys(json).length < 4) {
        return Object.entries(json)
          .map(([key, value]) => `${key}: ${value}`)
          .join(', ');
      }
      return JSON.stringify(json, null, 2);
    } catch (e) {
      return json.toString();
    }
  };

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #001440 0%, #002060 50%, #001440 100%)' }}>
      <style jsx>{`
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
          50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.6); }
        }
      `}</style>

      <div className="container mx-auto p-6 max-w-7xl">
        {/* Navigation */}
        <div className="mb-8">
          <Link href={`/workflows/${workflowId}`} 
            className="group inline-flex items-center px-6 py-3 rounded-2xl text-white transition-all duration-300"
            style={{
              background: 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
              border: '2px solid rgba(0, 212, 255, 0.4)',
              boxShadow: '0 4px 16px rgba(0, 180, 240, 0.3)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 212, 255, 0.5)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 180, 240, 0.3)';
            }}>
            <svg className="h-5 w-5 mr-3 group-hover:-translate-x-1 transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Retour aux tâches
          </Link>
        </div>

        {task && (
          <div className="space-y-8">
            {/* En-tête de la tâche */}
            <div className="backdrop-blur-xl rounded-3xl overflow-hidden border"
              style={{
                background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.8) 0%, rgba(0, 20, 64, 0.8) 100%)',
                borderColor: 'rgba(0, 180, 240, 0.3)',
                boxShadow: '0 8px 32px rgba(0, 32, 96, 0.5)',
              }}>
              <div className="px-8 py-6 relative overflow-hidden"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 180, 240, 0.15) 0%, rgba(0, 212, 255, 0.1) 100%)',
                  borderBottom: '2px solid rgba(0, 180, 240, 0.2)',
                }}>
                <div className="relative flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
                  <div>
                    <h1 className="text-3xl font-bold mb-2"
                      style={{
                        background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        letterSpacing: '0.5px',
                      }}>{task.name}</h1>
                    <p style={{ color: '#00B0F0', fontSize: '14px' }}>ID: {task.id}</p>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className={`px-5 py-3 rounded-2xl border backdrop-blur-sm ${getStatusClass(task.status).bg} ${getStatusClass(task.status).text} flex items-center`}
                      style={{ boxShadow: '0 4px 16px rgba(0, 180, 240, 0.2)' }}>
                      <div className={`w-2 h-2 rounded-full ${getStatusClass(task.status).dot} mr-3 animate-pulse`}></div>
                      <span className="mr-2 text-lg">{getStatusClass(task.status).icon}</span>
                      <span className="font-semibold">{task.status}</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="p-8">
                <div className="mb-8">
                  <div className="p-6 rounded-2xl border-l-4"
                    style={{
                      background: 'linear-gradient(135deg, rgba(0, 180, 240, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%)',
                      borderColor: '#00B0F0',
                      boxShadow: 'inset 0 2px 8px rgba(0, 180, 240, 0.1)',
                    }}>
                    <p className="text-lg leading-relaxed" style={{ color: '#FFFFFF' }}>{task.description}</p>
                  </div>
                </div>
                
                {/* Grille des détails */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-10">
                  {/* Détails de la tâche */}
                  <div className="p-6 rounded-2xl border backdrop-blur-sm"
                    style={{
                      background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
                      borderColor: 'rgba(0, 180, 240, 0.3)',
                      boxShadow: '0 4px 16px rgba(0, 32, 96, 0.3)',
                    }}>
                    <div className="flex items-center mb-6">
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center mr-4"
                        style={{
                          background: 'linear-gradient(135deg, rgba(0, 180, 240, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%)',
                          border: '2px solid rgba(0, 180, 240, 0.3)',
                        }}>
                        <svg className="h-6 w-6" style={{ color: '#00D4FF' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-bold" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>Détails de la tâche</h3>
                    </div>
                    <div className="space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center py-3"
                        style={{ borderBottom: '1px solid rgba(0, 180, 240, 0.2)' }}>
                        <span className="font-semibold w-40 mb-1 sm:mb-0" style={{ color: '#00B0F0' }}>Workflow:</span>
                        <span className="px-3 py-1 rounded-lg"
                          style={{
                            color: '#FFFFFF',
                            background: 'rgba(0, 180, 240, 0.1)',
                            border: '1px solid rgba(0, 180, 240, 0.2)',
                          }}>{task.workflow_name}</span>
                      </div>
                      <div className="flex flex-col sm:flex-row sm:items-center py-3"
                        style={{ borderBottom: '1px solid rgba(0, 180, 240, 0.2)' }}>
                        <span className="font-semibold w-40 mb-1 sm:mb-0" style={{ color: '#00B0F0' }}>Commande:</span>
                        <code className="px-4 py-2 rounded-lg font-mono text-sm"
                          style={{
                            background: 'rgba(0, 0, 0, 0.5)',
                            color: '#00D4FF',
                            border: '1px solid rgba(0, 180, 240, 0.2)',
                          }}>{task.command}</code>
                      </div>
                      <div className="py-3" style={{ borderBottom: '1px solid rgba(0, 180, 240, 0.2)' }}>
                        <span className="font-semibold block mb-2" style={{ color: '#00B0F0' }}>Paramètres:</span>
                        <pre className="p-4 rounded-xl text-sm overflow-x-auto font-mono"
                          style={{
                            background: 'rgba(0, 0, 0, 0.5)',
                            color: '#FFFFFF',
                            border: '1px solid rgba(0, 180, 240, 0.2)',
                          }}>{formatJSON(task.parameters)}</pre>
                      </div>
                      <div className="py-3" style={{ borderBottom: '1px solid rgba(0, 180, 240, 0.2)' }}>
                        <span className="font-semibold block mb-2" style={{ color: '#00B0F0' }}>Ressources requises:</span>
                        <pre className="p-4 rounded-xl text-sm overflow-x-auto font-mono"
                          style={{
                            background: 'rgba(0, 0, 0, 0.5)',
                            color: '#FFFFFF',
                            border: '1px solid rgba(0, 180, 240, 0.2)',
                          }}>{formatJSON(task.required_resources)}</pre>
                      </div>
                      <div className="flex flex-col sm:flex-row sm:items-center py-3"
                        style={{ borderBottom: '1px solid rgba(0, 180, 240, 0.2)' }}>
                        <span className="font-semibold w-40 mb-1 sm:mb-0" style={{ color: '#00B0F0' }}>Temps max estimé:</span>
                        <span className="px-3 py-1 rounded-lg"
                          style={{
                            color: '#FFA500',
                            background: 'rgba(255, 165, 0, 0.1)',
                            border: '1px solid rgba(255, 165, 0, 0.2)',
                          }}>{task.estimated_max_time} secondes</span>
                      </div>
                      <div className="flex flex-col sm:flex-row sm:items-center py-3"
                        style={{ borderBottom: '1px solid rgba(0, 180, 240, 0.2)' }}>
                        <span className="font-semibold w-40 mb-1 sm:mb-0" style={{ color: '#00B0F0' }}>Créé le:</span>
                        <span style={{ color: '#FFFFFF' }}>{new Date(task.created_at).toLocaleString('fr-FR')}</span>
                      </div>
                      {task.start_time && (
                        <div className="flex flex-col sm:flex-row sm:items-center py-3"
                          style={{ borderBottom: '1px solid rgba(0, 180, 240, 0.2)' }}>
                          <span className="font-semibold w-40 mb-1 sm:mb-0" style={{ color: '#00B0F0' }}>Démarré le:</span>
                          <span className="px-3 py-1 rounded-lg"
                            style={{
                              color: '#00FF00',
                              background: 'rgba(0, 255, 0, 0.1)',
                              border: '1px solid rgba(0, 255, 0, 0.2)',
                            }}>{new Date(task.start_time).toLocaleString('fr-FR')}</span>
                        </div>
                      )}
                      {task.end_time && (
                        <div className="flex flex-col sm:flex-row sm:items-center py-3">
                          <span className="font-semibold w-40 mb-1 sm:mb-0" style={{ color: '#00B0F0' }}>Terminé le:</span>
                          <span className="px-3 py-1 rounded-lg"
                            style={{
                              color: '#00D4FF',
                              background: 'rgba(0, 212, 255, 0.1)',
                              border: '1px solid rgba(0, 212, 255, 0.2)',
                            }}>{new Date(task.end_time).toLocaleString('fr-FR')}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Progression */}
                  <div className="p-6 rounded-2xl border backdrop-blur-sm"
                    style={{
                      background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
                      borderColor: 'rgba(0, 180, 240, 0.3)',
                      boxShadow: '0 4px 16px rgba(0, 32, 96, 0.3)',
                    }}>
                    <div className="flex items-center mb-6">
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center mr-4"
                        style={{
                          background: 'linear-gradient(135deg, rgba(0, 255, 0, 0.2) 0%, rgba(0, 200, 0, 0.1) 100%)',
                          border: '2px solid rgba(0, 255, 0, 0.3)',
                        }}>
                        <svg className="h-6 w-6" style={{ color: '#00FF00' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-bold" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>Progression</h3>
                    </div>
                    
                    <div className="mb-8">
                      <div className="flex justify-between mb-3">
                        <span className="text-sm font-semibold" style={{ color: '#00B0F0' }}>Avancement</span>
                        <span className="text-sm font-bold px-2 py-1 rounded-lg"
                          style={{
                            color: '#FFFFFF',
                            background: 'rgba(0, 180, 240, 0.2)',
                            border: '1px solid rgba(0, 180, 240, 0.3)',
                          }}>{task.progress}%</span>
                      </div>
                      <div className="w-full h-6 rounded-full overflow-hidden"
                        style={{
                          background: 'rgba(0, 0, 0, 0.3)',
                          boxShadow: 'inset 0 2px 8px rgba(0, 0, 0, 0.3)',
                        }}>
                        <div 
                          className="h-full rounded-full transition-all duration-700 ease-out"
                          style={{ 
                            width: `${task.progress}%`,
                            background: task.status === 'FAILED' 
                              ? 'linear-gradient(90deg, #FF0000 0%, #CC0000 100%)' 
                              : task.status === 'COMPLETED' 
                              ? 'linear-gradient(90deg, #00FF00 0%, #00CC00 100%)' 
                              : 'linear-gradient(90deg, #00B0F0 0%, #00D4FF 100%)',
                            boxShadow: '0 0 20px rgba(0, 212, 255, 0.5)',
                          }}
                        ></div>
                      </div>
                    </div>
                    
                    {task.start_time && task.end_time ? (
                      <div className="p-5 rounded-2xl border"
                        style={{
                          background: 'linear-gradient(135deg, rgba(0, 180, 240, 0.15) 0%, rgba(0, 212, 255, 0.1) 100%)',
                          borderColor: 'rgba(0, 180, 240, 0.3)',
                          boxShadow: 'inset 0 2px 8px rgba(0, 180, 240, 0.1)',
                        }}>
                        <h4 className="font-semibold mb-3 flex items-center" style={{ color: '#FFFFFF' }}>
                          <svg className="h-5 w-5 mr-2" style={{ color: '#00D4FF' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Durée d'exécution
                        </h4>
                        <div className="text-2xl font-bold" style={{ color: '#00D4FF' }}>
                          {((new Date(task.end_time).getTime() - new Date(task.start_time).getTime()) / 1000).toFixed(2)} secondes
                        </div>
                      </div>
                    ) : task.start_time ? (
                      <div className="p-5 rounded-2xl border"
                        style={{
                          background: 'linear-gradient(135deg, rgba(255, 165, 0, 0.15) 0%, rgba(255, 140, 0, 0.1) 100%)',
                          borderColor: 'rgba(255, 165, 0, 0.3)',
                          boxShadow: 'inset 0 2px 8px rgba(255, 165, 0, 0.1)',
                        }}>
                        <h4 className="font-semibold mb-3 flex items-center" style={{ color: '#FFFFFF' }}>
                          <svg className="h-5 w-5 mr-2 animate-pulse" style={{ color: '#FFA500' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          En cours depuis
                        </h4>
                        <div className="text-2xl font-bold" style={{ color: '#FFA500' }}>
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
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center mr-4"
                        style={{
                          background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.2) 0%, rgba(126, 34, 206, 0.1) 100%)',
                          border: '2px solid rgba(147, 51, 234, 0.3)',
                        }}>
                        <svg className="h-6 w-6" style={{ color: '#A78BFA' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-bold" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>
                        Volontaires assignés 
                        <span className="ml-3 px-3 py-1 rounded-full text-sm font-semibold"
                          style={{
                            background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.2) 0%, rgba(126, 34, 206, 0.1) 100%)',
                            color: '#A78BFA',
                            border: '1px solid rgba(147, 51, 234, 0.3)',
                          }}>
                          {volunteers.length}
                        </span>
                      </h3>
                    </div>
                    <button
                      onClick={() => setShowAssignForm(!showAssignForm)}
                      className="group inline-flex items-center px-6 py-3 rounded-2xl font-semibold transition-all duration-300"
                      style={{
                        background: 'linear-gradient(135deg, #00FF00 0%, #00CC00 100%)',
                        color: '#001440',
                        border: '2px solid rgba(0, 255, 0, 0.4)',
                        boxShadow: '0 4px 16px rgba(0, 255, 0, 0.3)',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = 'translateY(-2px)';
                        e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 255, 0, 0.5)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'translateY(0)';
                        e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 255, 0, 0.3)';
                      }}>
                      <svg className="h-5 w-5 mr-2 group-hover:rotate-90 transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                      </svg>
                      Assigner un volontaire
                    </button>
                  </div>
                  
                  {showAssignForm && (
                    <div className="p-6 rounded-2xl mb-8 border backdrop-blur-sm"
                      style={{
                        background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
                        borderColor: 'rgba(0, 180, 240, 0.3)',
                        boxShadow: '0 4px 16px rgba(0, 32, 96, 0.3)',
                      }}>
                      <h4 className="text-lg font-bold mb-4 flex items-center" style={{ color: '#FFFFFF' }}>
                        <svg className="h-5 w-5 mr-2" style={{ color: '#00FF00' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                        </svg>
                        Assigner un nouveau volontaire
                      </h4>
                      {availableVolunteers.length === 0 ? (
                        <div className="border p-5 rounded-2xl backdrop-blur-sm"
                          style={{
                            background: 'linear-gradient(135deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 140, 0, 0.05) 100%)',
                            borderColor: 'rgba(255, 165, 0, 0.3)',
                          }}>
                          <div className="flex items-center">
                            <svg className="h-6 w-6 mr-3 flex-shrink-0" style={{ color: '#FFA500' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                            <p className="font-medium" style={{ color: '#FFA500' }}>Aucun volontaire disponible pour cette tâche. Les volontaires doivent être disponibles et non déjà assignés.</p>
                          </div>
                        </div>
                      ) : (
                        <form onSubmit={handleAssignVolunteer} className="flex flex-col lg:flex-row lg:space-x-4 gap-4">
                          <div className="relative flex-grow">
                            <select
                              className="block w-full border-2 px-4 py-4 pr-10 rounded-2xl focus:outline-none focus:ring-2 transition-all duration-200"
                              style={{
                                background: 'rgba(0, 0, 0, 0.3)',
                                borderColor: 'rgba(0, 180, 240, 0.3)',
                                color: '#FFFFFF',
                              }}
                              value={selectedVolunteerId}
                              onChange={(e) => setSelectedVolunteerId(e.target.value)}
                              required
                              onFocus={(e) => {
                                e.currentTarget.style.borderColor = '#00D4FF';
                                e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.1)';
                              }}
                              onBlur={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                                e.currentTarget.style.boxShadow = 'none';
                              }}>
                              <option value="" style={{ background: '#001440', color: '#FFFFFF' }}>Sélectionnez un volontaire</option>
                              {availableVolunteers.map((volunteer) => (
                                <option key={volunteer.id} value={volunteer.id} style={{ background: '#001440', color: '#FFFFFF' }}>
                                  {volunteer.name} ({volunteer.hostname}) - {volunteer.cpu_cores} cœurs, {volunteer.ram_mb} MB RAM
                                </option>
                              ))}
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3" style={{ color: '#00B0F0' }}>
                              <svg className="fill-current h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                              </svg>
                            </div>
                          </div>
                          <div className="flex space-x-3">
                            <button
                              type="submit"
                              className="group flex-grow lg:flex-grow-0 inline-flex justify-center items-center px-6 py-4 rounded-2xl font-semibold focus:outline-none transition-all duration-200"
                              style={{
                                background: 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                                color: '#FFFFFF',
                                border: '2px solid rgba(0, 212, 255, 0.4)',
                                boxShadow: '0 4px 16px rgba(0, 180, 240, 0.3)',
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.transform = 'translateY(-2px)';
                                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 212, 255, 0.5)';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 180, 240, 0.3)';
                              }}>
                              <svg className="h-5 w-5 mr-2 group-hover:scale-110 transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                              </svg>
                              Assigner
                            </button>
                            <button
                              type="button"
                              onClick={() => setShowAssignForm(false)}
                              className="group flex-grow lg:flex-grow-0 inline-flex justify-center items-center px-6 py-4 rounded-2xl font-semibold focus:outline-none transition-all duration-200"
                              style={{
                                background: 'rgba(100, 100, 100, 0.2)',
                                color: '#FFFFFF',
                                border: '2px solid rgba(150, 150, 150, 0.3)',
                                boxShadow: '0 4px 16px rgba(100, 100, 100, 0.2)',
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.transform = 'translateY(-2px)';
                                e.currentTarget.style.background = 'rgba(150, 150, 150, 0.3)';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.background = 'rgba(100, 100, 100, 0.2)';
                              }}>
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
                  <div className="backdrop-blur-xl rounded-2xl p-6 border mb-8"
                    style={{
                      background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.8) 0%, rgba(0, 20, 64, 0.8) 100%)',
                      borderColor: 'rgba(0, 180, 240, 0.3)',
                      boxShadow: '0 8px 32px rgba(0, 32, 96, 0.5)',
                    }}>
                    <h3 className="text-xl font-bold mb-4" style={{ color: '#FFFFFF' }}>Détails des volontaires</h3>
                    <div className="overflow-x-auto">
                      <table className="min-w-full">
                        <thead>
                          <tr style={{ borderBottom: '2px solid rgba(0, 180, 240, 0.3)' }}>
                            <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#00B0F0' }}>Nom</th>
                            <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#00B0F0' }}>Ressources</th>
                            <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#00B0F0' }}>Statut</th>
                            <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#00B0F0' }}>Dernière activité</th>
                          </tr>
                        </thead>
                        <tbody>
                          {volunteers.map((v) => {
                            const status = getStatusClass(v.status);
                            return (
                              <tr key={v.id} className="transition-colors duration-200"
                                style={{ borderBottom: '1px solid rgba(0, 180, 240, 0.1)' }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.background = 'rgba(0, 180, 240, 0.05)';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.background = 'transparent';
                                }}>
                                <td className="px-6 py-4 text-sm font-medium" style={{ color: '#FFFFFF' }}>{v.name}</td>
                                <td className="px-6 py-4 text-sm" style={{ color: '#00B0F0' }}>
                                  {v.cpu_cores} cœurs, {v.ram_mb} MB RAM, {v.disk_gb} GB
                                </td>
                                <td className="px-6 py-4">
                                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${status.bg} ${status.text}`}>
                                    {status.icon} {v.status}
                                  </span>
                                </td>
                                <td className="px-6 py-4 text-sm" style={{ color: '#FFFFFF', opacity: 0.7 }}>
                                  {new Date(v.last_seen).toLocaleString('fr-FR')}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Sous-tâches */}
                {task.subtasks && task.subtasks.length > 0 && (
                  <div className="backdrop-blur-xl rounded-2xl p-6 border mb-8"
                    style={{
                      background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.8) 0%, rgba(0, 20, 64, 0.8) 100%)',
                      borderColor: 'rgba(0, 180, 240, 0.3)',
                      boxShadow: '0 8px 32px rgba(0, 32, 96, 0.5)',
                    }}>
                    <h3 className="text-xl font-bold mb-6" style={{ color: '#FFFFFF' }}>Sous-tâches</h3>
                    <ul className="space-y-4">
                      {task.subtasks.map((subtask) => {
                        const status = getStatusClass(subtask.status);
                        return (
                          <li key={subtask.id} className="p-4 rounded-xl border"
                            style={{
                              background: 'linear-gradient(135deg, rgba(0, 180, 240, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%)',
                              borderColor: 'rgba(0, 180, 240, 0.2)',
                            }}>
                            <div className="flex justify-between items-center">
                              <div>
                                <h4 className="font-semibold text-lg" style={{ color: '#FFFFFF' }}>{subtask.name}</h4>
                                {subtask.description && (
                                  <p className="text-sm mt-1" style={{ color: '#00B0F0' }}>{subtask.description}</p>
                                )}
                              </div>
                              <div className={`px-3 py-1 rounded-full text-xs font-semibold ${status.bg} ${status.text} flex items-center`}>
                                <span className="mr-2">{status.icon}</span> {subtask.status}
                              </div>
                            </div>
                            <div className="mt-3">
                              <div className="w-full rounded-full h-2.5"
                                style={{ background: 'rgba(0, 0, 0, 0.3)' }}>
                                <div
                                  className="h-full rounded-full"
                                  style={{
                                    width: `${subtask.progress}%`,
                                    background: subtask.status === 'FAILED'
                                      ? 'linear-gradient(90deg, #FF0000 0%, #CC0000 100%)'
                                      : subtask.status === 'COMPLETED'
                                      ? 'linear-gradient(90deg, #00FF00 0%, #00CC00 100%)'
                                      : 'linear-gradient(90deg, #00B0F0 0%, #00D4FF 100%)',
                                  }}
                                ></div>
                              </div>
                              <p className="text-sm mt-1" style={{ color: '#00B0F0' }}>{subtask.progress}%</p>
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                )}

                {/* Logs */}
                {task.logs && (
                  <div className="p-6 mt-10 rounded-2xl border"
                    style={{
                      background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.7) 0%, rgba(0, 0, 0, 0.5) 100%)',
                      borderColor: 'rgba(0, 180, 240, 0.3)',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
                    }}>
                    <h3 className="text-xl font-bold mb-4" style={{ color: '#00D4FF' }}>Logs d'exécution</h3>
                    <pre className="rounded-lg p-4 text-sm font-mono overflow-x-auto whitespace-pre-wrap"
                      style={{
                        background: 'rgba(0, 0, 0, 0.6)',
                        color: '#00FF00',
                        border: '1px solid rgba(0, 180, 240, 0.2)',
                      }}>{task.logs}</pre>
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