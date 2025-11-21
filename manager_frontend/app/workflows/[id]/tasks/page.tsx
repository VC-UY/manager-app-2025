"use client"

import { useEffect, useState } from 'react';
import { taskService } from '@/lib/api';
import Link from 'next/link';
import { Task } from '@/lib/types';

export default function TasksPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Couleurs selon la charte graphique
  const COLORS = {
    primary: '#3B82F6',
    secondary: '#60A5FA',
    background: '#0F172A',
    card: '#1E293B',
    border: '#334155',
    borderLight: '#475569',
    textPrimary: '#FFFFFF',
    textSecondary: '#94A3B8',
    textLight: '#CBD5E1',
  };

  useEffect(() => {
    const fetchTasks = async () => {
      try {
        setLoading(true);
        let tasksData;
        
        if (filterStatus) {
          tasksData = await taskService.getTasksByStatus(filterStatus);
        } else {
          tasksData = await taskService.getTasks();
        }
        
        setTasks(tasksData);
        setLoading(false);
      } catch (err: any) {
        console.error('Erreur lors du chargement des tâches:', err);
        setError(err.error || 'Une erreur est survenue lors du chargement des tâches');
        setLoading(false);
      }
    };

    fetchTasks();
  }, [filterStatus]);

  const getStatusInfo = (status: string) => {
    const statusMap: Record<string, any> = {
      'PENDING': { color: '#F59E0B', bg: '#F59E0B20', label: 'En attente' },
      'RUNNING': { color: COLORS.primary, bg: `${COLORS.primary}20`, label: 'En cours' },
      'COMPLETED': { color: '#10B981', bg: '#10B98120', label: 'Terminée' },
      'FAILED': { color: '#EF4444', bg: '#EF444420', label: 'Échouée' },
      'ASSIGNED': { color: '#8B5CF6', bg: '#8B5CF620', label: 'Assignée' }
    };
    return statusMap[status] || { color: COLORS.secondary, bg: `${COLORS.secondary}20`, label: status };
  };

  const filteredTasks = tasks.filter(task => {
    const matchesSearch = searchTerm === '' || 
      task.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      task.description?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === null || task.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="min-h-screen" style={{ 
      background: 'linear-gradient(135deg, #0A1628 0%, #1A2942 50%, #0A1628 100%)'
    }}>
      <style jsx>{`
        @keyframes slideIn { 
          from { opacity: 0; transform: translateY(20px); } 
          to { opacity: 1; transform: translateY(0); } 
        }
        @keyframes pulse { 
          0%, 100% { opacity: 1; transform: scale(1); } 
          50% { opacity: 0.8; transform: scale(1.05); } 
        }
        @keyframes spin { 
          to { transform: rotate(360deg); } 
        }
      `}</style>

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Top Navigation */}
        <div className="mb-6 p-4 rounded-2xl backdrop-blur-xl"
          style={{
            background: `${COLORS.card}CC`,
            border: `1px solid ${COLORS.border}`,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
            animation: 'slideIn 0.6s ease-out'
          }}>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Link href="/workflows" 
                className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm transition-all duration-300"
                style={{
                  background: `${COLORS.border}60`,
                  color: COLORS.textPrimary,
                  border: `1px solid ${COLORS.border}`,
                }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
                Workflows
              </Link>
              
              <Link href="/tasks" 
                className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm transition-all duration-300"
                style={{
                  background: `linear-gradient(135deg, ${COLORS.primary} 0%, ${COLORS.secondary} 100%)`,
                  color: COLORS.textPrimary,
                  border: `1px solid ${COLORS.primary}`,
                }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                Tâches
              </Link>
              
              <Link href="/volunteers" 
                className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm transition-all duration-300"
                style={{
                  background: `${COLORS.border}60`,
                  color: COLORS.textPrimary,
                  border: `1px solid ${COLORS.border}`,
                }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
                Volontaires
              </Link>
            </div>
            
            <Link href="/profile"
              className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm transition-all duration-300"
              style={{
                background: `${COLORS.primary}30`,
                color: COLORS.textPrimary,
                border: `1px solid ${COLORS.primary}50`,
              }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
              Profil
            </Link>
          </div>
        </div>

        {/* Header avec stats */}
        <div className="relative mb-8 p-8 rounded-3xl backdrop-blur-xl overflow-hidden"
          style={{
            background: `${COLORS.card}E6`,
            border: `1px solid ${COLORS.border}`,
            boxShadow: `0 0 40px ${COLORS.primary}20`,
            animation: 'slideIn 0.8s ease-out'
          }}>
          {/* Effet de fond animé */}
          <div className="absolute top-0 right-0 w-96 h-96 rounded-full opacity-10"
            style={{
              background: `radial-gradient(circle, ${COLORS.secondary} 0%, transparent 70%)`,
              animation: 'pulse 4s ease-in-out infinite'
            }} />
          
          <div className="relative z-10">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h1 className="text-4xl font-bold mb-3" style={{
                  background: `linear-gradient(135deg, ${COLORS.textPrimary} 0%, ${COLORS.secondary} 100%)`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  letterSpacing: '0.5px'
                }}>
                  Toutes les Tâches
                </h1>
                <p style={{ color: COLORS.textSecondary, fontSize: '16px', maxWidth: '600px' }}>
                  Consultez et gérez toutes les tâches de vos workflows distribués. Suivez leur progression et assignez-les à des volontaires.
                </p>
              </div>
              <div className="hidden md:block w-16 h-16 rounded-2xl flex items-center justify-center"
                style={{
                  background: `linear-gradient(135deg, ${COLORS.primary}40 0%, ${COLORS.secondary}20 100%)`,
                  border: `1px solid ${COLORS.primary}50`,
                }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke={COLORS.secondary} strokeWidth="2">
                  <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[
                { 
                  label: 'En attente', 
                  value: tasks.filter(t => t.status === 'PENDING').length, 
                  color: '#F59E0B',
                  icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z'
                },
                { 
                  label: 'En cours', 
                  value: tasks.filter(t => t.status === 'RUNNING').length, 
                  color: COLORS.primary,
                  icon: 'M13 10V3L4 14h7v7l9-11h-7z'
                },
                { 
                  label: 'Terminées', 
                  value: tasks.filter(t => t.status === 'COMPLETED').length, 
                  color: '#10B981',
                  icon: 'M5 13l4 4L19 7'
                },
                { 
                  label: 'Échouées', 
                  value: tasks.filter(t => t.status === 'FAILED').length, 
                  color: '#EF4444',
                  icon: 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'
                }
              ].map((stat, idx) => (
                <div key={idx} className="p-4 rounded-xl backdrop-blur-sm transition-all duration-300"
                  style={{
                    background: `${stat.color}15`,
                    border: `1px solid ${stat.color}30`
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-4px)';
                    e.currentTarget.style.boxShadow = `0 8px 24px ${stat.color}40`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}>
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl flex items-center justify-center"
                      style={{ 
                        background: `${stat.color}25`, 
                        border: `1px solid ${stat.color}50` 
                      }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={stat.color} strokeWidth="2">
                        <path d={stat.icon} />
                      </svg>
                    </div>
                    <div>
                      <div style={{ color: COLORS.textSecondary, fontSize: '12px', fontWeight: 500 }}>
                        {stat.label}
                      </div>
                      <div style={{ color: COLORS.textPrimary, fontSize: '24px', fontWeight: 700 }}>
                        {stat.value}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Filtres et recherche */}
        <div className="mb-6 p-4 rounded-2xl backdrop-blur-xl"
          style={{
            background: `${COLORS.card}CC`,
            border: `1px solid ${COLORS.border}`
          }}>
          <div className="flex flex-col md:flex-row gap-4">
            {/* Barre de recherche */}
            <div className="flex-1 relative">
              <svg className="absolute left-3 top-1/2 transform -translate-y-1/2" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={COLORS.textSecondary} strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <path d="m21 21-4.35-4.35" />
              </svg>
              <input
                type="text"
                placeholder="Rechercher une tâche..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 rounded-xl transition-all duration-300"
                style={{
                  background: `${COLORS.border}40`,
                  border: `1px solid ${COLORS.border}`,
                  color: COLORS.textPrimary,
                  outline: 'none'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = COLORS.primary;
                  e.target.style.boxShadow = `0 0 20px ${COLORS.primary}30`;
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = COLORS.border;
                  e.target.style.boxShadow = 'none';
                }}
              />
            </div>
            
            {/* Filtre statut */}
            <select
              value={filterStatus || ''}
              onChange={(e) => setFilterStatus(e.target.value === '' ? null : e.target.value)}
              className="px-4 py-3 rounded-xl transition-all duration-300"
              style={{
                background: `${COLORS.border}40`,
                border: `1px solid ${COLORS.border}`,
                color: COLORS.textPrimary,
                outline: 'none'
              }}>
              <option value="">Tous les statuts</option>
              <option value="PENDING">En attente</option>
              <option value="RUNNING">En cours</option>
              <option value="COMPLETED">Terminée</option>
              <option value="FAILED">Échouée</option>
              <option value="ASSIGNED">Assignée</option>
            </select>
          </div>
        </div>

        {/* Message d'erreur */}
        {error && (
          <div className="mb-6 p-4 rounded-xl" style={{
            background: '#EF444420',
            border: '1px solid #EF444450',
            animation: 'slideIn 0.5s ease-out'
          }}>
            <div className="flex items-center gap-2">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#EF4444" strokeWidth="2">
                <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p style={{ color: '#EF4444', fontSize: '14px', fontWeight: 500 }}>{error}</p>
            </div>
          </div>
        )}

        {/* Loading */}
        {loading ? (
          <div className="flex flex-col items-center justify-center p-12 rounded-2xl backdrop-blur-xl"
            style={{
              background: `${COLORS.card}CC`,
              border: `1px solid ${COLORS.border}`
            }}>
            <div className="w-16 h-16 rounded-full mb-4"
              style={{
                border: `4px solid ${COLORS.border}`,
                borderTopColor: COLORS.primary,
                animation: 'spin 1s linear infinite'
              }} />
            <p style={{ color: COLORS.textSecondary, fontSize: '16px', fontWeight: 500 }}>
              Chargement des tâches...
            </p>
            <p style={{ color: COLORS.textSecondary, fontSize: '14px', marginTop: '8px', opacity: 0.7 }}>
              Merci de patienter un instant
            </p>
          </div>
        ) : filteredTasks.length === 0 ? (
          <div className="text-center p-12 rounded-2xl backdrop-blur-xl"
            style={{
              background: `${COLORS.card}CC`,
              border: `1px solid ${COLORS.border}`
            }}>
            <div className="w-20 h-20 mx-auto mb-6 rounded-2xl flex items-center justify-center"
              style={{
                background: `${COLORS.primary}20`,
                border: `1px solid ${COLORS.primary}40`
              }}>
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={COLORS.secondary} strokeWidth="2">
                <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <h2 style={{ color: COLORS.textPrimary, fontSize: '24px', fontWeight: 700, marginBottom: '12px' }}>
              Aucune tâche trouvée
            </h2>
            <p style={{ color: COLORS.textSecondary, fontSize: '14px', marginBottom: '24px', maxWidth: '500px', margin: '0 auto 24px' }}>
              Vous n'avez pas encore de tâches. Les tâches sont créées lorsque vous soumettez un workflow ou que vous les créez manuellement.
            </p>
            <Link href="/workflows"
              className="inline-flex items-center gap-2 px-8 py-3 rounded-xl font-bold transition-all duration-300"
              style={{
                background: `linear-gradient(135deg, ${COLORS.primary} 0%, ${COLORS.secondary} 100%)`,
                color: COLORS.textPrimary,
                border: `1px solid ${COLORS.primary}`
              }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
              Voir les workflows
            </Link>
          </div>
        ) : (
          <div className="rounded-2xl backdrop-blur-xl overflow-hidden"
            style={{
              background: `${COLORS.card}CC`,
              border: `1px solid ${COLORS.border}`
            }}>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead style={{ background: `${COLORS.border}40` }}>
                  <tr>
                    {['Nom', 'Workflow', 'Statut', 'Progression', 'Volontaires', 'Actions'].map((header, idx) => (
                      <th key={idx} className="px-6 py-4 text-left text-xs font-semibold uppercase tracking-wider"
                        style={{ color: COLORS.textSecondary }}>
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredTasks.map((task) => {
                    const statusInfo = getStatusInfo(task.status);
                    return (
                      <tr key={task.id} 
                        className="transition-all duration-200"
                        style={{ 
                          borderBottom: `1px solid ${COLORS.border}40`,
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = `${COLORS.border}20`;
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = 'transparent';
                        }}>
                        <td className="px-6 py-4">
                          <div>
                            <div style={{ color: COLORS.textPrimary, fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>
                              {task.name}
                            </div>
                            <div style={{ color: COLORS.textSecondary, fontSize: '12px' }}>
                              {task.description?.substring(0, 50)}{task.description && task.description.length > 50 ? '...' : ''}
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div style={{ color: COLORS.textPrimary, fontSize: '13px', marginBottom: '4px' }}>
                            {task.workflow_name || '-'}
                          </div>
                          {task.workflow && (
                            <Link href={`/workflows/${task.workflow}`} 
                              style={{ color: COLORS.primary, fontSize: '12px' }}
                              className="hover:underline">
                              Voir le workflow
                            </Link>
                          )}
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={{ 
                              background: statusInfo.bg, 
                              color: statusInfo.color,
                              border: `1px solid ${statusInfo.color}50`
                            }}>
                            {statusInfo.label}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="w-full rounded-full h-2"
                            style={{ background: `${COLORS.border}60` }}>
                            <div className="h-2 rounded-full transition-all duration-500"
                              style={{ 
                                width: `${task.progress}%`,
                                background: `linear-gradient(90deg, ${COLORS.primary} 0%, ${COLORS.secondary} 100%)`
                              }} />
                          </div>
                          <p style={{ color: COLORS.textSecondary, fontSize: '11px', marginTop: '4px' }}>
                            {task.progress || 0}%
                          </p>
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={{ 
                              background: '#8B5CF620', 
                              color: '#8B5CF6',
                              border: '1px solid #8B5CF650'
                            }}>
                            {task.volunteer_count || 0} volontaire(s)
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex gap-2">
                            <Link href={`/tasks/${task.id}`}
                              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200"
                              style={{
                                background: `${COLORS.primary}20`,
                                color: COLORS.primary,
                                border: `1px solid ${COLORS.primary}40`
                              }}>
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                <path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                              </svg>
                              Détails
                            </Link>
                            <Link href={`/tasks/${task.id}/volunteers`}
                              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200"
                              style={{
                                background: '#8B5CF620',
                                color: '#8B5CF6',
                                border: '1px solid #8B5CF650'
                              }}>
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                              </svg>
                              Volontaires
                            </Link>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}