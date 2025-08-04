'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { workflowService } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';
import { ProfileModal } from '@/components/ProfileModal';

// Types
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
  
  // État pour filtrer et trier les workflows
  const [filterStatus, setFilterStatus] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<string>('created_at');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    console.log("État d'authentification:", isAuthenticated);
    
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
        console.error('Erreur lors du chargement des workflows:', err);
        setError(err.message || 'Une erreur est survenue lors du chargement des workflows');
        
        // Données exemple en cas d'erreur
        setWorkflows([
          {
            id: 'test-1',
            name: 'Analyse des données clients',
            description: 'Ce workflow analyse les données clients pour générer des insights marketing et des tendances de consommation',
            workflow_type: 'ANALYTICS',
            status: 'RUNNING',
            created_at: new Date().toISOString()
          },
          {
            id: 'test-2',
            name: 'Génération de rapports mensuels',
            description: 'Automatisation de la création et distribution des rapports mensuels de performance',
            workflow_type: 'REPORTING',
            status: 'COMPLETED',
            created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()
          },
          {
            id: 'test-3',
            name: 'Import des données fournisseurs',
            description: 'Workflow d\'intégration et de normalisation des données fournisseurs dans notre système',
            workflow_type: 'INTEGRATION',
            status: 'FAILED',
            created_at: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString()
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflows();
    
    const fallbackTimer = setTimeout(() => {
      if (loading) {
        setLoading(false);
        setError("Le chargement a pris trop de temps. Veuillez rafraîchir la page.");
      }
    }, 15000);
    
    return () => clearTimeout(fallbackTimer);
  }, [isAuthenticated]);



  const handleDelete = async (id: string) => {
    if (!confirm("Êtes-vous sûr de vouloir supprimer ce workflow ?")) return;
    try {
      setLoading(true);
      await workflowService.deleteWorkflow(id);
      // on filtre le workflow supprimé de la liste
      setWorkflows((ws) => ws.filter((w) => w.id !== id));
    } catch (err: any) {
      console.error("Erreur suppression :", err);
      setError(err.error || "Échec de la suppression du workflow");
    } finally {
      setLoading(false);
    }
  };

  // Fonction pour filtrer les workflows
  const filteredWorkflows = workflows.filter(workflow => {
    // Filtre par recherche (nom ou description)
    const matchesSearch = searchTerm === '' || 
      workflow.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      workflow.description.toLowerCase().includes(searchTerm.toLowerCase());
    
    // Filtre par statut
    const matchesStatus = filterStatus === null || workflow.status === filterStatus;
    
    return matchesSearch && matchesStatus;
  }).sort((a, b) => {
    // Tri des workflows
    if (sortBy === 'created_at') {
      return sortDirection === 'asc' 
        ? new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        : new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    } else if (sortBy === 'name') {
      return sortDirection === 'asc'
        ? a.name.localeCompare(b.name)
        : b.name.localeCompare(a.name);
    }
    return 0;
  });

  // Fonction pour formater la date
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }).format(date);
    } catch (e) {
      return dateString;
    }
  };

  // Fonction pour obtenir la couleur et l'icône du badge selon le statut
  const getStatusInfo = (status: string) => {
    switch (status) {
      case 'CREATED':
        return {
          bgColor: 'bg-gray-200',
          textColor: 'text-gray-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          ),
          label: 'Créé'
        };
      case 'VALIDATED':
        return {
          bgColor: 'bg-blue-100',
          textColor: 'text-blue-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ),
          label: 'Validé'
        };
      case 'SUBMITTED':
        return {
          bgColor: 'bg-yellow-100',
          textColor: 'text-yellow-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          ),
          label: 'Soumis'
        };
      case 'RUNNING':
        return {
          bgColor: 'bg-green-100',
          textColor: 'text-green-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          ),
          label: 'En cours'
        };
      case 'COMPLETED':
        return {
          bgColor: 'bg-green-200',
          textColor: 'text-green-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          ),
          label: 'Terminé'
        };
      case 'FAILED':
        return {
          bgColor: 'bg-red-100',
          textColor: 'text-red-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ),
          label: 'Échoué'
        };
      default:
        return {
          bgColor: 'bg-gray-100',
          textColor: 'text-gray-800',
          icon: null,
          label: status
        };
    }
  };

  // Obtenir les icônes et informations sur les types de workflows
  const getWorkflowTypeInfo = (type: string) => {
    switch (type) {
      case 'ANALYTICS':
        return {
          bgColor: 'bg-purple-100',
          textColor: 'text-purple-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          ),
          description: "Analyse de données et génération d'insights"
        };
      case 'REPORTING':
        return {
          bgColor: 'bg-indigo-100',
          textColor: 'text-indigo-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          ),
          description: "Génération automatique de rapports et tableaux de bord"
        };
      case 'INTEGRATION':
        return {
          bgColor: 'bg-blue-100',
          textColor: 'text-blue-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
          ),
          description: "Intégration et transfert de données entre systèmes"
        };
      case 'TEST':
        return {
          bgColor: 'bg-gray-100',
          textColor: 'text-gray-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          ),
          description: "Tests et vérifications de processus"
        };
      default:
        return {
          bgColor: 'bg-purple-100',
          textColor: 'text-purple-800',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
            </svg>
          ),
          description: "Workflow spécialisé"
        };
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
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
              
              <Link 
                href="/volunteers" 
                className="inline-flex items-center px-4 py-2 bg-indigo-800 hover:bg-indigo-700 text-white font-medium rounded-lg transition-colors duration-200"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
                Volontaires
              </Link>
            </div>
            
            <div className="flex items-center space-x-3 mt-2 sm:mt-0">
              <span className="text-white text-sm hidden md:inline-block">Plateforme de gestion de workflows distribués</span>
              <ProfileModal />
            </div>
          </div>
        </div>
      </div>

      {/* Header avec animation et dégradé */}
      <div className="relative bg-gradient-to-r from-blue-700 to-indigo-800 rounded-xl shadow-lg mb-8 overflow-hidden">
        <div className="absolute inset-0 bg-grid-white/10 opacity-10"></div>
        <div className="relative z-10 px-8 py-10 text-white">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold mb-2">Mes Workflows</h1>
              <p className="text-blue-100 max-w-2xl">
                Gérez et suivez tous vos workflows dans un environnement centralisé. Créez, modifiez et surveillez l'état de vos processus automatisés.
              </p>
            </div>
            <div className="hidden md:block">
              <div className="relative p-3 bg-white/10 rounded-full backdrop-blur-sm">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
                </svg>
              </div>
            </div>
          </div>
          
          {/* Stats cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 border border-white/20">
              <div className="flex items-center">
                <div className="bg-blue-500 p-2 rounded-md mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div>
                  <div className="text-sm text-blue-100">Workflows actifs</div>
                  <div className="text-xl font-semibold">
                    {workflows.filter(w => w.status === 'RUNNING').length}
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 border border-white/20">
              <div className="flex items-center">
                <div className="bg-green-500 p-2 rounded-md mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div>
                  <div className="text-sm text-blue-100">Terminés</div>
                  <div className="text-xl font-semibold">
                    {workflows.filter(w => w.status === 'COMPLETED').length}
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 border border-white/20">
              <div className="flex items-center">
                <div className="bg-red-500 p-2 rounded-md mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <div className="text-sm text-blue-100">Échoués</div>
                  <div className="text-xl font-semibold">
                    {workflows.filter(w => w.status === 'FAILED').length}
                  </div>
                </div>
              </div>
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

      {/* Barre d'action et filtres */}
      <div className="bg-white rounded-xl shadow-md mb-6 p-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-3 md:space-y-0">
          <div className="flex flex-grow max-w-md">
            <div className="relative flex-grow">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
                </svg>
              </div>
              <input
                type="text"
                placeholder="Rechercher un workflow..."
                className="focus:ring-indigo-500 focus:border-indigo-500 block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <select
              className="rounded-md border-gray-300 py-2 pl-3 pr-10 text-base focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              value={filterStatus || ''}
              onChange={(e) => setFilterStatus(e.target.value === '' ? null : e.target.value)}
            >
              <option value="">Tous les statuts</option>
              <option value="CREATED">Créé</option>
              <option value="VALIDATED">Validé</option>
              <option value="SUBMITTED">Soumis</option>
              <option value="RUNNING">En cours</option>
              <option value="COMPLETED">Terminé</option>
              <option value="FAILED">Échoué</option>
            </select>
            
            <Link
              href="/workflows/create"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              Créer un workflow
            </Link>
          </div>
        </div>
      </div>

      {/* Message d'erreur avec animation */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 mb-6 rounded-md shadow-sm animate-fadeIn">
          <div className="flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
           </svg>
           <span className="font-medium">{error}</span>
         </div>
       </div>
     )}

     {/* État de chargement amélioré */}
     {loading ? (
       <div className="flex flex-col justify-center items-center h-64 bg-white rounded-xl shadow-md p-8">
         <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500 mb-4"></div>
         <p className="text-gray-600 text-lg animate-pulse">Chargement de vos workflows...</p>
         <p className="text-gray-500 text-sm mt-2">Merci de patienter un instant</p>
       </div>
     ) : (
       <>
         {workflows.length === 0 ? (
           <div className="bg-white rounded-xl p-12 text-center shadow-md border border-gray-100">
             <svg xmlns="http://www.w3.org/2000/svg" className="h-20 w-20 mx-auto text-gray-400 mb-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
             </svg>
             <h2 className="text-2xl font-medium text-gray-700 mb-3">Aucun workflow trouvé</h2>
             <p className="text-gray-500 max-w-md mx-auto mb-6">
               Vous n'avez pas encore créé de workflow. Les workflows vous permettent d'automatiser vos processus métier et d'améliorer votre productivité.
             </p>
             <Link
               href="/workflows/create"
               className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-md text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all duration-200"
             >
               <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
               </svg>
               Créer votre premier workflow
             </Link>
           </div>
         ) : (
           <>
             {/* Nombre de résultats */}
             <div className="mb-4 text-sm text-gray-500">
               {filteredWorkflows.length} workflows trouvés
               {filterStatus && <span> avec le statut {filterStatus}</span>}
               {searchTerm && <span> contenant "{searchTerm}"</span>}
             </div>
             
             {/* Liste de workflows en cards pour les petits écrans */}
             <div className="grid grid-cols-1 gap-4 md:hidden">
               {filteredWorkflows.map((workflow) => {
                 const statusInfo = getStatusInfo(workflow.status);
                 const typeInfo = getWorkflowTypeInfo(workflow.workflow_type);
                 
                 return (
                   <div key={workflow.id} className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100 hover:shadow-lg transition-shadow duration-200">
                     <div className="p-4">
                       <div className="flex justify-between items-start">
                         <div>
                           <Link 
                             href={`/workflows/${workflow.id}`} 
                             className="text-lg font-medium text-blue-600 hover:text-blue-800 hover:underline mb-1 block"
                           >
                             {workflow.name}
                           </Link>
                           <p className="text-sm text-gray-500 mb-2">{workflow.description.substring(0, 100)}{workflow.description.length > 100 ? '...' : ''}</p>
                           
                           <div className="flex flex-wrap gap-2 mt-3">
                             <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.bgColor} ${statusInfo.textColor}`}>
                               {statusInfo.icon}
                               {statusInfo.label}
                             </span>
                             
                             <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${typeInfo.bgColor} ${typeInfo.textColor}`}>
                               {typeInfo.icon}
                               {workflow.workflow_type}
                             </span>
                             
                             <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                               <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                               </svg>
                               {formatDate(workflow.created_at)}
                             </span>
                           </div>
                         </div>
                       </div>
                       
                       <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-gray-100">
                         <Link 
                           href={`/workflows/${workflow.id}`} 
                           className="text-indigo-600 hover:text-indigo-900 text-sm font-medium flex items-center"
                         >
                           <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                           </svg>
                           Détails
                         </Link>
                         
                         
                         <Link 
                           href={`/workflows/${workflow.id}/edit`} 
                           className="text-yellow-600 hover:text-yellow-800 text-sm font-medium flex items-center"
                         >
                           <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                           </svg>
                           Modifier
                         </Link>
                         
                         <button
                           onClick={() => handleDelete(workflow.id)}
                           className="text-red-600 hover:text-red-800 text-sm font-medium flex items-center"
                         >
                           <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                           </svg>
                           Supprimer
                         </button>
                       </div>
                     </div>
                   </div>
                 );
               })}
             </div>
             
             {/* Tableau pour les écrans moyens et grands */}
             <div className="hidden md:block">
               <div className="bg-white shadow-md rounded-xl overflow-hidden border border-gray-200">
                 <div className="overflow-x-auto">
                   <table className="min-w-full divide-y divide-gray-200">
                     <thead className="bg-gray-50">
                       <tr>
                         <th 
                           scope="col" 
                           className="group px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                           onClick={() => {
                             if (sortBy === 'name') {
                               setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
                             } else {
                               setSortBy('name');
                               setSortDirection('asc');
                             }
                           }}
                         >
                           <div className="flex items-center">
                             Nom
                             {sortBy === 'name' && (
                               <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                 {sortDirection === 'asc' ? (
                                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                                 ) : (
                                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                 )}
                               </svg>
                             )}
                           </div>
                         </th>
                         <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                           Type
                         </th>
                         <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                           Statut
                         </th>
                         <th 
                           scope="col" 
                           className="group px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                           onClick={() => {
                             if (sortBy === 'created_at') {
                               setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
                             } else {
                               setSortBy('created_at');
                               setSortDirection('desc');
                             }
                           }}
                         >
                           <div className="flex items-center">
                             Date de création
                             {sortBy === 'created_at' && (
                               <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                 {sortDirection === 'asc' ? (
                                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                                 ) : (
                                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                 )}
                               </svg>
                             )}
                           </div>
                         </th>
                         <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                           Actions
                         </th>
                       </tr>
                     </thead>
                     <tbody className="bg-white divide-y divide-gray-200">
                       {filteredWorkflows.map((workflow) => {
                         const statusInfo = getStatusInfo(workflow.status);
                         const typeInfo = getWorkflowTypeInfo(workflow.workflow_type);
                         
                         return (
                           <tr key={workflow.id} className="hover:bg-gray-50 transition-colors duration-150">
                             <td className="px-6 py-4">
                               <Link 
                                 href={`/workflows/${workflow.id}`} 
                                 className="text-blue-600 hover:text-blue-800 font-medium hover:underline block"
                               >
                                 {workflow.name}
                               </Link>
                               <p className="text-sm text-gray-500 mt-1 line-clamp-2">{workflow.description}</p>
                             </td>
                             <td className="px-6 py-4 whitespace-nowrap">
                               <div className="flex flex-col">
                                 <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${typeInfo.bgColor} ${typeInfo.textColor}`}>
                                   {typeInfo.icon}
                                   {workflow.workflow_type}
                                 </span>
                                 <span className="text-xs text-gray-500 mt-1">{typeInfo.description}</span>
                               </div>
                             </td>
                             <td className="px-6 py-4 whitespace-nowrap">
                               <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.bgColor} ${statusInfo.textColor}`}>
                                 {statusInfo.icon}
                                 {statusInfo.label}
                               </span>
                             </td>
                             <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                               <div className="flex items-center">
                                 <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                 </svg>
                                 {formatDate(workflow.created_at)}
                               </div>
                             </td>
                             <td className="px-6 py-4 whitespace-nowrap">
                               <div className="flex flex-wrap gap-2">
                                 <Link 
                                   href={`/workflows/${workflow.id}`} 
                                   className="inline-flex items-center px-2.5 py-1.5 bg-indigo-50 border border-indigo-200 rounded-md text-xs font-medium text-indigo-600 hover:bg-indigo-100 transition-colors duration-200"
                                 >
                                   <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                     <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                     <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                   </svg>
                                   Détails
                                 </Link>
                                
                                 
                                 <Link 
                                   href={`/workflows/${workflow.id}/edit`} 
                                   className="inline-flex items-center px-2.5 py-1.5 bg-yellow-50 border border-yellow-200 rounded-md text-xs font-medium text-yellow-700 hover:bg-yellow-100 transition-colors duration-200"
                                 >
                                   <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                     <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                   </svg>
                                   Modifier
                                 </Link>
                                 
                                 <button
                                   onClick={() => handleDelete(workflow.id)}
                                   className="inline-flex items-center px-2.5 py-1.5 bg-red-50 border border-red-200 rounded-md text-xs font-medium text-red-600 hover:bg-red-100 transition-colors duration-200"
                                 >
                                   <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                     <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                   </svg>
                                   Supprimer
                                 </button>
                               </div>
                             </td>
                           </tr>
                         );
                       })}
                     </tbody>
                   </table>
                 </div>
               </div>
             </div>
             
             {/* Pagination */}
             {filteredWorkflows.length > 0 && (
               <div className="flex justify-between items-center mt-6 bg-white p-4 rounded-lg shadow-sm">
                 <div className="text-sm text-gray-500">
                   Affichage de <span className="font-medium">{filteredWorkflows.length}</span> workflows
                 </div>
                 <div className="flex space-x-2">
                   <button className="px-3 py-1 border border-gray-300 rounded-md text-sm text-gray-500 hover:bg-gray-50 disabled:opacity-50" disabled>
                     Précédent
                   </button>
                   <button className="px-3 py-1 border border-gray-300 rounded-md text-sm bg-blue-50 text-blue-600 font-medium border-blue-200">
                     1
                   </button>
                   <button className="px-3 py-1 border border-gray-300 rounded-md text-sm text-gray-500 hover:bg-gray-50 disabled:opacity-50" disabled>
                     Suivant
                   </button>
                 </div>
               </div>
             )}
           </>
         )}
       </>
     )}
     
     {/* Section informative au bas de la page avec fond plus contrasté */}
     <div className="mt-12 bg-slate-50 rounded-xl p-6 border border-slate-200 shadow-sm">
       <h2 className="text-lg font-medium text-slate-800 mb-4">À propos des Workflows</h2>
       <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
         <div className="flex">
           <div className="flex-shrink-0">
             <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-600 text-white">
               <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
               </svg>
             </div>
           </div>
           <div className="ml-4">
             <h3 className="text-lg font-medium text-slate-900">Automatisation</h3>
             <p className="mt-2 text-sm text-slate-600">
               Automatisez vos processus récurrents pour gagner du temps et réduire les erreurs manuelles.
             </p>
           </div>
         </div>
         
         <div className="flex">
           <div className="flex-shrink-0">
             <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-600 text-white">
               <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
               </svg>
             </div>
           </div>
           <div className="ml-4">
             <h3 className="text-lg font-medium text-slate-900">Suivi en temps réel</h3>
             <p className="mt-2 text-sm text-slate-600">
               Surveillez l'état de vos workflows et recevez des notifications sur leur progression.
             </p>
           </div>
         </div>
         
         <div className="flex">
           <div className="flex-shrink-0">
             <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-600 text-white">
               <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
               </svg>
             </div>
           </div>
           <div className="ml-4">
             <h3 className="text-lg font-medium text-slate-900">Intégration</h3>
             <p className="mt-2 text-sm text-slate-600">
               Connectez vos workflows à d'autres systèmes et applications pour une expérience fluide.
             </p>
           </div>
         </div>
       </div>
     </div>
   </div>
 );
}