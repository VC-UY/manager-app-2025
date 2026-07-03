'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { workflowService } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';
import { ProfileModal } from '@/components/ProfileModal';

interface Workflow {
  id: string;
  name: string;
  description: string;
  workflow_type: string;
  status: string;
  created_at: string;
}

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { isAuthenticated } = useAuth();
  
  const [filterStatus, setFilterStatus] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<string>('created_at');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    const fetchWorkflows = async () => {
      try {
        setLoading(true);
        
        if (!isAuthenticated) {
          setError("Veuillez vous connecter pour voir vos workflows");
          setLoading(false);
          return;
        }
        
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Le chargement a pris trop de temps')), 10000)
        );
        
        const data = await Promise.race([
          workflowService.getWorkflows(),
          timeoutPromise
        ]);
        
        if (Array.isArray(data)) {
          setWorkflows(data);
          setError(null);
        }
      } catch (err: any) {
        console.error('Erreur:', err);
        setError(err.message || 'Une erreur est survenue');
        setWorkflows([]);
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflows();
  }, [isAuthenticated]);

  const handleDelete = async (id: string) => {
    if (!confirm("Êtes-vous sûr de vouloir supprimer ce workflow ?")) return;
    try {
      setLoading(true);
      await workflowService.deleteWorkflow(id);
      setWorkflows((ws) => ws.filter((w) => w.id !== id));
    } catch (err: any) {
      setError(err.error || "Échec de la suppression");
    } finally {
      setLoading(false);
    }
  };

  const filteredWorkflows = workflows.filter(workflow => {
    const matchesSearch = searchTerm === '' || 
      workflow.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      workflow.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === null || workflow.status === filterStatus;
    return matchesSearch && matchesStatus;
  }).sort((a, b) => {
    if (sortBy === 'created_at') {
      return sortDirection === 'asc' 
        ? new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        : new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    } else if (sortBy === 'name') {
      return sortDirection === 'asc' ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
    }
    return 0;
  });

  const formatDate = (dateString: string) => {
    try {
      return new Intl.DateTimeFormat('fr-FR', {
        day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'
      }).format(new Date(dateString));
    } catch (e) {
      return dateString;
    }
  };

  const getStatusInfo = (status: string) => {
    const statusMap: Record<string, any> = {
      'CREATED': { color: '#00B0F0', bg: 'rgba(0, 180, 240, 0.1)', label: 'Créé' },
      'VALIDATED': { color: '#00D4FF', bg: 'rgba(0, 212, 255, 0.1)', label: 'Validé' },
      'SUBMITTED': { color: '#00B0F0', bg: 'rgba(0, 180, 240, 0.1)', label: 'Soumis' },
      'RUNNING': { color: '#00D4FF', bg: 'rgba(0, 212, 255, 0.15)', label: 'En cours' },
      'COMPLETED': { color: '#00B0F0', bg: 'rgba(0, 180, 240, 0.1)', label: 'Terminé' },
      'FAILED': { color: '#FF4444', bg: 'rgba(255, 68, 68, 0.1)', label: 'Échoué' }
    };
    return statusMap[status] || { color: '#00B0F0', bg: 'rgba(0, 180, 240, 0.1)', label: status };
  };

  const getWorkflowTypeInfo = (type: string) => {
    const typeMap: Record<string, any> = {
      'ANALYTICS': { color: '#00D4FF', bg: 'rgba(0, 212, 255, 0.1)' },
      'REPORTING': { color: '#00B0F0', bg: 'rgba(0, 180, 240, 0.1)' },
      'INTEGRATION': { color: '#00D4FF', bg: 'rgba(0, 212, 255, 0.1)' },
      'TEST': { color: '#00B0F0', bg: 'rgba(0, 180, 240, 0.1)' }
    };
    return typeMap[type] || { color: '#00D4FF', bg: 'rgba(0, 212, 255, 0.1)' };
  };

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #001440 0%, #002060 50%, #001440 100%)' }}>
      <style jsx>{`
        @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Top Navigation */}
        <div className="mb-6 p-4 rounded-2xl backdrop-blur-xl"
          style={{
            background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.8) 0%, rgba(0, 20, 64, 0.8) 100%)',
            border: '2px solid rgba(0, 180, 240, 0.3)',
            boxShadow: '0 8px 32px rgba(0, 32, 96, 0.5)',
          }}>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              {[
                { href: '/workflows', label: 'Workflows', active: true },
                { href: '/tasks', label: 'Tasks', active: false },
                { href: '/volunteers', label: 'Volontaires', active: false }
              ].map((item, idx) => (
                <Link key={idx} href={item.href}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm transition-all duration-300"
                  style={{
                    background: item.active ? 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)' : 'rgba(255, 255, 255, 0.1)',
                    color: '#FFFFFF',
                    border: item.active ? '2px solid rgba(0, 212, 255, 0.4)' : '2px solid rgba(0, 180, 240, 0.2)',
                    letterSpacing: '0.3px'
                  }}>
                  {item.label}
                </Link>
              ))}
            </div>
            <ProfileModal />
          </div>
        </div>

        {/* Header */}
        <div className="relative mb-8 p-8 rounded-3xl backdrop-blur-xl overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.9) 0%, rgba(0, 20, 64, 0.9) 100%)',
            border: '2px solid rgba(0, 180, 240, 0.3)',
            boxShadow: '0 12px 48px rgba(0, 32, 96, 0.6)',
            animation: 'slideIn 0.8s ease-out'
          }}>
          <div className="absolute top-0 right-0 w-64 h-64 rounded-full opacity-10"
            style={{
              background: 'radial-gradient(circle, rgba(0, 212, 255, 0.3) 0%, transparent 70%)',
              animation: 'pulse 4s ease-in-out infinite'
            }} />
          
          <div className="relative z-10">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h1 className="text-4xl font-bold mb-3" style={{
                  background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  letterSpacing: '0.5px'
                }}>
                  Mes Workflows
                </h1>
                <p style={{ color: '#00B0F0', fontSize: '16px' }}>
                  Gérez et suivez tous vos workflows de calcul distribué
                </p>
              </div>
              <div className="hidden md:block w-16 h-16 rounded-2xl flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 180, 240, 0.1) 100%)',
                  border: '2px solid rgba(0, 212, 255, 0.3)',
                  animation: 'float 3s ease-in-out infinite'
                }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00D4FF" strokeWidth="2">
                  <path d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
                </svg>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                { label: 'Actifs', value: workflows.filter(w => w.status === 'RUNNING').length, color: '#00D4FF', icon: 'M13 10V3L4 14h7v7l9-11h-7z' },
                { label: 'Terminés', value: workflows.filter(w => w.status === 'COMPLETED').length, color: '#00B0F0', icon: 'M5 13l4 4L19 7' },
                { label: 'Échoués', value: workflows.filter(w => w.status === 'FAILED').length, color: '#FF4444', icon: 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' }
              ].map((stat, idx) => (
                <div key={idx} className="p-4 rounded-xl backdrop-blur-sm transition-all duration-300"
                  style={{
                    background: 'rgba(0, 180, 240, 0.1)',
                    border: '2px solid rgba(0, 180, 240, 0.2)'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = stat.color;
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.2)';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}>
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl flex items-center justify-center"
                      style={{ background: `${stat.color}20`, border: `2px solid ${stat.color}40` }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={stat.color} strokeWidth="2">
                        <path d={stat.icon} />
                      </svg>
                    </div>
                    <div>
                      <div style={{ color: '#00B0F0', fontSize: '12px', fontWeight: 500 }}>{stat.label}</div>
                      <div style={{ color: '#FFFFFF', fontSize: '24px', fontWeight: 700 }}>{stat.value}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Search & Filters */}
        <div className="mb-6 p-4 rounded-2xl backdrop-blur-xl"
          style={{
            background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
            border: '2px solid rgba(0, 180, 240, 0.3)'
          }}>
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <svg className="absolute left-3 top-1/2 transform -translate-y-1/2" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00B0F0" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <path d="m21 21-4.35-4.35" />
              </svg>
              <input
                type="text"
                placeholder="Rechercher un workflow..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 rounded-xl transition-all duration-300"
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '2px solid rgba(0, 180, 240, 0.3)',
                  color: '#FFFFFF',
                  outline: 'none'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = '#00D4FF';
                  e.target.style.boxShadow = '0 0 20px rgba(0, 212, 255, 0.3)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                  e.target.style.boxShadow = 'none';
                }}
              />
            </div>
            
            <select
              value={filterStatus || ''}
              onChange={(e) => setFilterStatus(e.target.value === '' ? null : e.target.value)}
              className="px-4 py-3 rounded-xl transition-all duration-300"
              style={{
                background: 'rgba(255, 255, 255, 0.1)',
                border: '2px solid rgba(0, 180, 240, 0.3)',
                color: '#FFFFFF',
                outline: 'none'
              }}>
              <option value="">Tous les statuts</option>
              <option value="CREATED">Créé</option>
              <option value="RUNNING">En cours</option>
              <option value="COMPLETED">Terminé</option>
              <option value="FAILED">Échoué</option>
            </select>
            
            <Link href="/workflows/create"
              className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold transition-all duration-300"
              style={{
                background: 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                color: '#FFFFFF',
                border: '2px solid rgba(0, 212, 255, 0.4)',
                boxShadow: '0 4px 16px rgba(0, 180, 240, 0.3)',
                letterSpacing: '0.3px'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 212, 255, 0.5)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 180, 240, 0.3)';
              }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 5v14M5 12h14" />
              </svg>
              Créer
            </Link>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 rounded-xl" style={{
            background: 'linear-gradient(90deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 68, 68, 0.05) 100%)',
            border: '2px solid rgba(255, 68, 68, 0.3)',
            animation: 'slideIn 0.5s ease-out'
          }}>
            <p style={{ color: '#FF8888', fontSize: '14px', fontWeight: 500 }}>{error}</p>
          </div>
        )}

        {/* Loading */}
        {loading ? (
          <div className="flex flex-col items-center justify-center p-12 rounded-2xl backdrop-blur-xl"
            style={{
              background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
              border: '2px solid rgba(0, 180, 240, 0.3)'
            }}>
            <div className="w-16 h-16 rounded-full mb-4"
              style={{
                border: '4px solid rgba(0, 180, 240, 0.2)',
                borderTopColor: '#00D4FF',
                animation: 'spin 1s linear infinite'
              }} />
            <p style={{ color: '#00B0F0', fontSize: '16px', fontWeight: 500 }}>Chargement...</p>
          </div>
        ) : workflows.length === 0 ? (
          <div className="text-center p-12 rounded-2xl backdrop-blur-xl"
            style={{
              background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
              border: '2px solid rgba(0, 180, 240, 0.3)'
            }}>
            <div className="w-20 h-20 mx-auto mb-6 rounded-2xl flex items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 180, 240, 0.1) 100%)',
                border: '2px solid rgba(0, 212, 255, 0.3)'
              }}>
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#00D4FF" strokeWidth="2">
                <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <h2 style={{ color: '#FFFFFF', fontSize: '24px', fontWeight: 700, marginBottom: '12px' }}>
              Aucun workflow
            </h2>
            <p style={{ color: '#00B0F0', fontSize: '14px', marginBottom: '24px' }}>
              Créez votre premier workflow pour commencer
            </p>
            <Link href="/workflows/create"
              className="inline-flex items-center gap-2 px-8 py-3 rounded-xl font-bold transition-all duration-300"
              style={{
                background: 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                color: '#FFFFFF',
                border: '2px solid rgba(0, 212, 255, 0.4)'
              }}>
              Créer un workflow
            </Link>
          </div>
        ) : (
          <>
            <div style={{ color: '#00B0F0', fontSize: '13px', marginBottom: '16px' }}>
              {filteredWorkflows.length} workflow(s) trouvé(s)
            </div>

            {/* Workflows Grid */}
            <div className="grid grid-cols-1 gap-4">
              {filteredWorkflows.map((workflow) => {
                const statusInfo = getStatusInfo(workflow.status);
                const typeInfo = getWorkflowTypeInfo(workflow.workflow_type);
                
                return (
                  <div key={workflow.id}
                    className="p-6 rounded-2xl backdrop-blur-xl transition-all duration-300"
                    style={{
                      background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
                      border: '2px solid rgba(0, 180, 240, 0.3)'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = '#00D4FF';
                      e.currentTarget.style.transform = 'translateY(-4px)';
                      e.currentTarget.style.boxShadow = '0 12px 40px rgba(0, 212, 255, 0.3)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = 'none';
                    }}>
                    
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                      <div className="flex-1">
                        <Link href={`/workflows/${workflow.id}`}
                          style={{ color: '#00D4FF', fontSize: '18px', fontWeight: 700, letterSpacing: '0.3px' }}
                          className="hover:underline">
                          {workflow.name}
                        </Link>
                        <p style={{ color: '#00B0F0', fontSize: '14px', marginTop: '8px' }}>
                          {workflow.description}
                        </p>
                        
                        <div className="flex flex-wrap gap-2 mt-4">
                          <span className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={{ background: statusInfo.bg, color: statusInfo.color, border: `1px solid ${statusInfo.color}40` }}>
                            {statusInfo.label}
                          </span>
                          <span className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={{ background: typeInfo.bg, color: typeInfo.color, border: `1px solid ${typeInfo.color}40` }}>
                            {workflow.workflow_type}
                          </span>
                          <span className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={{ background: 'rgba(0, 180, 240, 0.1)', color: '#00B0F0', border: '1px solid rgba(0, 180, 240, 0.3)' }}>
                            {formatDate(workflow.created_at)}
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex gap-2">
                        <Link href={`/workflows/${workflow.id}`}
                          className="px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300"
                          style={{
                            background: 'rgba(0, 212, 255, 0.1)',
                            color: '#00D4FF',
                            border: '2px solid rgba(0, 212, 255, 0.3)'
                          }}>
                          Voir
                        </Link>
                        <Link href={`/workflows/${workflow.id}/edit`}
                          className="px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300"
                          style={{
                            background: 'rgba(0, 180, 240, 0.1)',
                            color: '#00B0F0',
                            border: '2px solid rgba(0, 180, 240, 0.3)'
                          }}>
                          Modifier
                        </Link>
                        <button
                          onClick={() => handleDelete(workflow.id)}
                          className="px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300"
                          style={{
                            background: 'rgba(255, 68, 68, 0.1)',
                            color: '#FF4444',
                            border: '2px solid rgba(255, 68, 68, 0.3)'
                          }}>
                          Supprimer
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}