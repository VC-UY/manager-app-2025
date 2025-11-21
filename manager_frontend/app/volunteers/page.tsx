// app/volunteers/page.tsx
"use client"

import { useEffect, useState } from 'react';
import { volunteerService } from '@/lib/api';
import { Volunteer } from '../../lib/types';
import Link from 'next/link';
import { ProfileModal } from '@/components/ProfileModal';

export default function VolunteersPage() {
  const [volunteers, setVolunteers] = useState<Volunteer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        let volunteerData;
        if (filter === 'all') {
          volunteerData = await volunteerService.getVolunteers();
        } else {
          volunteerData = await volunteerService.getVolunteersByStatus(filter);
        }
        
        setVolunteers(volunteerData);
        setLoading(false);
      } catch (err: any) {
        setError(err.error || 'Une erreur est survenue');
        setLoading(false);
      }
    };

    fetchData();
  }, [filter]);

  const filteredVolunteers = volunteers.filter(volunteer => {
    const matchesSearch = searchTerm === '' || 
      volunteer.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      volunteer.hostname.toLowerCase().includes(searchTerm.toLowerCase()) ||
      volunteer.ip_address.includes(searchTerm);
    return matchesSearch;
  });

  const getStatusInfo = (status: string) => {
    const statusMap: Record<string, any> = {
      'AVAILABLE': { color: '#00D4FF', bg: 'rgba(0, 212, 255, 0.15)', label: 'Disponible', icon: '🟢' },
      'BUSY': { color: '#00B0F0', bg: 'rgba(0, 180, 240, 0.15)', label: 'Occupé', icon: '🔵' },
      'OFFLINE': { color: '#888888', bg: 'rgba(136, 136, 136, 0.15)', label: 'Hors ligne', icon: '⚪' },
      'MAINTENANCE': { color: '#FFA500', bg: 'rgba(255, 165, 0, 0.15)', label: 'Maintenance', icon: '🟠' }
    };
    return statusMap[status] || { color: '#00B0F0', bg: 'rgba(0, 180, 240, 0.1)', label: status, icon: '❔' };
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
                { href: '/workflows', label: 'Workflows', active: false },
                { href: '/tasks', label: 'Tasks', active: false },
                { href: '/volunteers', label: 'Volontaires', active: true }
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
                  Volontaires
                </h1>
                <p style={{ color: '#00B0F0', fontSize: '16px' }}>
                  Gérez et surveillez tous vos nœuds de calcul distribués
                </p>
              </div>
              <div className="hidden md:block w-16 h-16 rounded-2xl flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 180, 240, 0.1) 100%)',
                  border: '2px solid rgba(0, 212, 255, 0.3)',
                  animation: 'float 3s ease-in-out infinite'
                }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00D4FF" strokeWidth="2">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                  <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
                </svg>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[
                { label: 'Total', value: volunteers.length, color: '#00D4FF', icon: 'M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M9 7a4 4 0 1 0 0-8 4 4 0 0 0 0 8z' },
                { label: 'Disponibles', value: volunteers.filter(v => v.status === 'AVAILABLE').length, color: '#00FF88', icon: 'M5 13l4 4L19 7' },
                { label: 'Occupés', value: volunteers.filter(v => v.status === 'BUSY').length, color: '#00B0F0', icon: 'M13 10V3L4 14h7v7l9-11h-7z' },
                { label: 'Hors ligne', value: volunteers.filter(v => v.status === 'OFFLINE').length, color: '#888888', icon: 'M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 0 1-3.46 0' }
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
                placeholder="Rechercher un volontaire..."
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
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="px-4 py-3 rounded-xl transition-all duration-300"
              style={{
                background: 'rgba(255, 255, 255, 0.1)',
                border: '2px solid rgba(0, 180, 240, 0.3)',
                color: '#FFFFFF',
                outline: 'none'
              }}>
              <option value="all">Tous les statuts</option>
              <option value="AVAILABLE">Disponibles</option>
              <option value="BUSY">Occupés</option>
              <option value="OFFLINE">Hors ligne</option>
              <option value="MAINTENANCE">En maintenance</option>
            </select>
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
        ) : filteredVolunteers.length === 0 ? (
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
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            </div>
            <h2 style={{ color: '#FFFFFF', fontSize: '24px', fontWeight: 700, marginBottom: '12px' }}>
              Aucun volontaire trouvé
            </h2>
            <p style={{ color: '#00B0F0', fontSize: '14px' }}>
              Essayez de modifier vos filtres de recherche
            </p>
          </div>
        ) : (
          <>
            <div style={{ color: '#00B0F0', fontSize: '13px', marginBottom: '16px' }}>
              {filteredVolunteers.length} volontaire(s) trouvé(s)
            </div>

            {/* Volunteers Grid */}
            <div className="grid grid-cols-1 gap-4">
              {filteredVolunteers.map((volunteer) => {
                const statusInfo = getStatusInfo(volunteer.status);
                
                return (
                  <div key={volunteer.id}
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
                    
                    <div className="flex flex-col lg:flex-row lg:items-center gap-6">
                      {/* Volunteer Info */}
                      <div className="flex-1">
                        <div className="flex items-center gap-4 mb-4">
                          <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-xl font-bold"
                            style={{
                              background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 180, 240, 0.1) 100%)',
                              border: '2px solid rgba(0, 212, 255, 0.3)',
                              color: '#00D4FF'
                            }}>
                            {volunteer.name.charAt(0).toUpperCase()}
                          </div>
                          <div>
                            <h3 style={{ color: '#00D4FF', fontSize: '20px', fontWeight: 700, letterSpacing: '0.3px' }}>
                              {volunteer.name}
                            </h3>
                            <div className="flex flex-wrap items-center gap-2 mt-1">
                              <span style={{ color: '#00B0F0', fontSize: '14px' }}>{volunteer.hostname}</span>
                              <span style={{ color: 'rgba(0, 180, 240, 0.5)' }}>•</span>
                              <span style={{ color: '#00B0F0', fontSize: '14px' }}>{volunteer.ip_address}</span>
                            </div>
                          </div>
                        </div>

                        {/* Resources */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                          <div className="p-3 rounded-xl" style={{ background: 'rgba(0, 180, 240, 0.1)', border: '1px solid rgba(0, 180, 240, 0.2)' }}>
                            <div style={{ color: '#00B0F0', fontSize: '11px', marginBottom: '4px' }}>CPU Cores</div>
                            <div style={{ color: '#FFFFFF', fontSize: '16px', fontWeight: 700 }}>{volunteer.cpu_cores}</div>
                          </div>
                          <div className="p-3 rounded-xl" style={{ background: 'rgba(0, 180, 240, 0.1)', border: '1px solid rgba(0, 180, 240, 0.2)' }}>
                            <div style={{ color: '#00B0F0', fontSize: '11px', marginBottom: '4px' }}>RAM</div>
                            <div style={{ color: '#FFFFFF', fontSize: '16px', fontWeight: 700 }}>{volunteer.ram_mb} MB</div>
                          </div>
                          <div className="p-3 rounded-xl" style={{ background: 'rgba(0, 180, 240, 0.1)', border: '1px solid rgba(0, 180, 240, 0.2)' }}>
                            <div style={{ color: '#00B0F0', fontSize: '11px', marginBottom: '4px' }}>Disque</div>
                            <div style={{ color: '#FFFFFF', fontSize: '16px', fontWeight: 700 }}>{volunteer.disk_gb} GB</div>
                          </div>
                          {volunteer.gpu && (
                            <div className="p-3 rounded-xl" style={{ background: 'rgba(0, 212, 255, 0.15)', border: '1px solid rgba(0, 212, 255, 0.3)' }}>
                              <div style={{ color: '#00D4FF', fontSize: '11px', marginBottom: '4px' }}>GPU</div>
                              <div style={{ color: '#FFFFFF', fontSize: '16px', fontWeight: 700 }}>{volunteer.gpu}</div>
                            </div>
                          )}
                        </div>

                        {/* Tags */}
                        <div className="flex flex-wrap gap-2">
                          <span className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={{ background: statusInfo.bg, color: statusInfo.color, border: `1px solid ${statusInfo.color}40` }}>
                            {statusInfo.icon} {statusInfo.label}
                          </span>
                          <span className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={{ 
                              background: volunteer.available ? 'rgba(0, 255, 136, 0.15)' : 'rgba(255, 165, 0, 0.15)', 
                              color: volunteer.available ? '#00FF88' : '#FFA500',
                              border: volunteer.available ? '1px solid rgba(0, 255, 136, 0.3)' : '1px solid rgba(255, 165, 0, 0.3)'
                            }}>
                            {volunteer.available ? '✓ Disponible' : '✗ Non disponible'}
                          </span>
                          <span className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={{ background: 'rgba(138, 43, 226, 0.15)', color: '#BA7FFF', border: '1px solid rgba(138, 43, 226, 0.3)' }}>
                            {volunteer.assigned_tasks_count || 0} tâche(s)
                          </span>
                          <span className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={{ background: 'rgba(0, 180, 240, 0.1)', color: '#00B0F0', border: '1px solid rgba(0, 180, 240, 0.3)' }}>
                            {new Date(volunteer.last_seen).toLocaleString('fr-FR', {
                              day: '2-digit',
                              month: '2-digit',
                              year: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit'
                            })}
                          </span>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex lg:flex-col gap-2">
                        <Link href={`/volunteers/${volunteer.id}`}
                          className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 whitespace-nowrap"
                          style={{
                            background: 'rgba(0, 212, 255, 0.1)',
                            color: '#00D4FF',
                            border: '2px solid rgba(0, 212, 255, 0.3)'
                          }}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                          </svg>
                          Détails
                        </Link>
                        <Link href={`/volunteers/${volunteer.id}/tasks`}
                          className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 whitespace-nowrap"
                          style={{
                            background: 'rgba(0, 180, 240, 0.1)',
                            color: '#00B0F0',
                            border: '2px solid rgba(0, 180, 240, 0.3)'
                          }}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                          </svg>
                          Tâches
                        </Link>
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