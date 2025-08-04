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

  if (loading) {
    return (
      <div className="c">
        <div className="container mx-auto p-6">
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-black">
        <div className="container mx-auto p-6">
          <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 rounded-lg shadow-md">
            <div className="flex items-center">
              <svg className="h-6 w-6 text-red-500 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <p className="font-medium">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const statusColors = {
    AVAILABLE: {
      bg: 'bg-emerald-100',
      text: 'text-emerald-800',
      border: 'border-emerald-200',
      icon: '🟢'
    },
    BUSY: {
      bg: 'bg-blue-100',
      text: 'text-blue-800',
      border: 'border-blue-200',
      icon: '🔵'
    },
    OFFLINE: {
      bg: 'bg-slate-100',
      text: 'text-slate-800',
      border: 'border-slate-200',
      icon: '⚪'
    },
    MAINTENANCE: {
      bg: 'bg-amber-100',
      text: 'text-amber-800',
      border: 'border-amber-200',
      icon: '🟠'
    }
  };

  return (
    <div className="min-h-screen bg-black">
        {/* Barre de navigation supérieure */}
      <div className="bg-gradient-to-r from-blue-800 to-indigo-900 rounded-lg shadow-md mb-6 overflow-hidden">
        <div className="px-4 py-3">
          <div className="flex flex-wrap items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link 
                href="/workflows" 
                className="inline-flex items-center px-4 py-2 bg-blue-700 hover:bg-blue-600 text-white font-medium rounded-lg transition-colors duration-200"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
                Workflows
              </Link>
              
              <Link 
                href="/tasks" 
                className="inline-flex items-center px-4 py-2 bg-indigo-800 hover:bg-indigo-700 text-white font-medium rounded-lg transition-colors duration-200"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                Tasks
              </Link>
              
        
            </div>
            
            <div className="flex items-center space-x-3 mt-2 sm:mt-0">
              <span className="text-white text-sm hidden md:inline-block">Plateforme de gestion de workflows distribués</span>
             <ProfileModal />
            </div>
          </div>
        </div>
      </div>
      <div className="container mx-auto p-6">
        <div className="flex flex-col md:flex-row justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-indigo-900 mb-4 md:mb-0">
            Volontaires
            <span className="ml-2 text-base font-normal text-indigo-600 bg-indigo-100 px-3 py-1 rounded-full">
              {volunteers.length} au total
            </span>
          </h1>
          
          <div className="w-full md:w-auto">
            <div className="relative">
              <select
                className="appearance-none block w-full md:w-56 bg-white border border-indigo-200 hover:border-indigo-500 px-4 py-3 pr-8 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-indigo-900 font-medium transition-all duration-300"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              >
                <option value="all">Tous les statuts</option>
                <option value="AVAILABLE">Disponibles</option>
                <option value="BUSY">Occupés</option>
                <option value="OFFLINE">Hors ligne</option>
                <option value="MAINTENANCE">En maintenance</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-indigo-700">
                <svg className="fill-current h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                  <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        <div className="overflow-hidden bg-white rounded-xl shadow-lg border border-indigo-100">
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="bg-indigo-600 text-white">
                  <th className="px-6 py-4 text-left text-sm font-medium uppercase tracking-wider">
                    Nom
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium uppercase tracking-wider">
                    Ressources
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium uppercase tracking-wider">
                    Statut
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium uppercase tracking-wider">
                    Tâches
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium uppercase tracking-wider">
                    Dernière activité
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-indigo-100">
                {volunteers.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-10 text-center">
                      <div className="flex flex-col items-center">
                        <svg className="w-16 h-16 text-indigo-200 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        <p className="text-lg font-medium text-indigo-900">Aucun volontaire disponible</p>
                        <p className="text-indigo-500 mt-1">Essayez de modifier vos filtres ou revenez plus tard</p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  volunteers.map((volunteer) => {
                    const statusStyle = statusColors[volunteer.status as keyof typeof statusColors] || {
                      bg: 'bg-gray-200',
                      text: 'text-gray-800',
                      border: 'border-gray-300',
                      icon: '❔'
                    };
                    
                    
                    return (
                      <tr key={volunteer.id} className="hover:bg-indigo-50 transition-colors duration-150">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="flex-shrink-0 h-10 w-10 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-700 font-bold">
                              {volunteer.name.charAt(0).toUpperCase()}
                            </div>
                            <div className="ml-4">
                              <div className="text-lg font-medium text-indigo-900">{volunteer.name}</div>
                              <div className="flex flex-col sm:flex-row sm:items-center mt-1">
                                <div className="text-indigo-700">{volunteer.hostname}</div>
                                <span className="hidden sm:block mx-2 text-indigo-300">•</span>
                                <div className="text-indigo-500 text-sm">{volunteer.ip_address}</div>
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex flex-col space-y-1">
                            <div className="inline-flex items-center">
                              <svg className="h-4 w-4 text-indigo-500 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
                              </svg>
                              <span className="text-indigo-900 font-medium">{volunteer.cpu_cores} CPU cores</span>
                            </div>
                            <div className="inline-flex items-center">
                              <svg className="h-4 w-4 text-indigo-500 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                              </svg>
                              <span className="text-indigo-800">{volunteer.ram_mb} MB RAM</span>
                            </div>
                            <div className="inline-flex items-center">
                              <svg className="h-4 w-4 text-indigo-500 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                              </svg>
                              <span className="text-indigo-800">{volunteer.disk_gb} GB Disk</span>
                            </div>
                            {volunteer.gpu && (
                              <div className="inline-flex items-center">
                                <svg className="h-4 w-4 text-indigo-500 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                </svg>
                                <span className="text-indigo-800 font-medium">GPU: {volunteer.gpu}</span>
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${statusStyle.bg} ${statusStyle.text}`}>
                            <span className="mr-1">{statusStyle.icon}</span>
                            {volunteer.status}
                          </div>
                          <p className={`mt-2 text-sm ${volunteer.available ? 'text-emerald-600' : 'text-amber-600'}`}>
                            {volunteer.available ? '✓ Disponible' : '✗ Non disponible'}
                          </p>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-purple-100 text-purple-800">
                            <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                            </svg>
                            {volunteer.assigned_tasks_count || 0} tâche(s)
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <svg className="h-4 w-4 text-indigo-500 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <span className="text-indigo-800">
                              {new Date(volunteer.last_seen).toLocaleString('fr-FR', {
                                day: '2-digit',
                                month: '2-digit',
                                year: 'numeric',
                                hour: '2-digit',
                                minute: '2-digit'
                              })}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <div className="flex flex-col sm:flex-row gap-2">
                            <Link 
                              href={`/volunteers/${volunteer.id}`} 
                              className="inline-flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-150"
                            >
                              <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                              </svg>
                              Détails
                            </Link>
                            <Link 
                              href={`/volunteers/${volunteer.id}/tasks`} 
                              className="inline-flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-indigo-700 bg-indigo-100 hover:bg-indigo-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-150"
                            >
                              <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                              </svg>
                              Tâches
                            </Link>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}