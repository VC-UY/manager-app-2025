import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Box, Grid, Typography, Card, Divider, Button, Chip,
  IconButton, LinearProgress, Backdrop, CircularProgress,
  Alert, Snackbar
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { 
  PlayArrow, Pause, Refresh, Loop, Timeline, Storage, 
  CloudDownload, Assessment, Share, Delete, ArrowBack
} from '@mui/icons-material';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import { 
  fetchWorkflowDetails, 
  submitWorkflow, 
  pauseWorkflow, 
  resumeWorkflow,
  cancelWorkflow
} from '../services/api';
import coordinationService from '../services/CoordinationService';

// Composants stylisés personnalisés
const GlassCard = styled(Card)(({ theme }) => ({
  background: 'rgba(255, 255, 255, 0.95)',
  backdropFilter: 'blur(10px)',
  borderRadius: 16,
  boxShadow: '0 8px 32px rgba(10, 36, 99, 0.15)',
  overflow: 'visible',
  position: 'relative',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: -4,
    left: 20,
    right: 20,
    height: 8,
    background: theme.palette.primary.main,
    borderRadius: '4px 4px 0 0',
  }
}));

const ActionButton = styled(Button)(({ theme, color }) => ({
  borderRadius: 12,
  padding: '10px 24px',
  background: color || theme.palette.primary.main,
  '&:hover': {
    background: color ? `${color}dd` : theme.palette.primary.dark,
    boxShadow: '0 6px 16px rgba(0, 0, 0, 0.2)',
    transform: 'translateY(-2px)',
  },
  transition: 'all 0.3s ease',
}));

const StatsCard = styled(Box)(({ theme, accentColor }) => ({
  background: '#fff',
  borderRadius: 12,
  padding: theme.spacing(2),
  position: 'relative',
  overflow: 'hidden',
  height: '100%',
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 3,
    background: accentColor || theme.palette.primary.main
  }
}));

// Noeud personnalisé pour ReactFlow
const TaskNode = ({ data }) => {
  const statusColor = getStatusColor(data.status);
  
  return (
    <Box 
      sx={{
        background: '#fff',
        border: `2px solid ${statusColor}`,
        borderRadius: 2,
        padding: 2,
        width: 200,
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
      }}
    >
      <Typography variant="subtitle2" fontWeight="bold">{data.label}</Typography>
      <Chip 
        label={data.status} 
        size="small"
        sx={{
          backgroundColor: statusColor,
          color: '#fff',
          fontSize: '0.7rem',
          my: 0.5
        }}
      />
      {data.progress !== undefined && (
        <LinearProgress 
          variant="determinate" 
          value={data.progress} 
          sx={{ height: 5, borderRadius: 5, mt: 1 }}
        />
      )}
    </Box>
  );
};

// Couleur basée sur le statut
const getStatusColor = (status) => {
  switch (status) {
    case 'RUNNING': return '#4CAF50';
    case 'COMPLETED': return '#2196F3';
    case 'FAILED': return '#D90429';
    case 'PAUSED': return '#FFC107';
    case 'PENDING': return '#FF9800';
    case 'SUBMITTED': return '#03A9F4';
    case 'SPLITTING': return '#00BCD4';
    case 'ASSIGNING': return '#009688';
    case 'PARTIAL_FAILURE': return '#FF5722';
    case 'REASSIGNING': return '#795548';
    case 'AGGREGATING': return '#607D8B';
    default: return '#9E9E9E';
  }
};

// Calcul de progression basé sur les tâches
const calculateProgress = (tasks) => {
  if (!tasks || tasks.length === 0) return 0;
  
  const completedTasks = tasks.filter(task => task.status === 'COMPLETED').length;
  return Math.round((completedTasks / tasks.length) * 100);
};

// Calcul des statistiques de tâches
const calculateTaskStats = (tasks) => {
  if (!tasks || tasks.length === 0) {
    return { completed: 0, running: 0, pending: 0 };
  }
  
  return {
    completed: tasks.filter(task => task.status === 'COMPLETED').length,
    running: tasks.filter(task => ['RUNNING', 'ASSIGNING', 'SPLITTING'].includes(task.status)).length,
    pending: tasks.filter(task => ['PENDING', 'SUBMITTED'].includes(task.status)).length
  };
};

// Composant principal
const WorkflowDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [workflow, setWorkflow] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [executionTime, setExecutionTime] = useState('--');
  const [remainingTime, setRemainingTime] = useState('--');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  // Chargement initial des données
  useEffect(() => {
    loadWorkflowData();
    
    // S'abonner aux mises à jour du statut du workflow
    const unsubscribe = coordinationService.subscribeToWorkflowStatus(id, (data) => {
      // Mettre à jour le workflow si des données de statut sont reçues
      if (data && data.status) {
        loadWorkflowData();
      }
    });
    
    // Nettoyage à la déconnexion
    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, [id]);

  // Fonction pour charger les données du workflow
  const loadWorkflowData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await fetchWorkflowDetails(id);
      setWorkflow(data);
      
      // Calculer le temps d'exécution si disponible
      if (data.submitted_at) {
        const submitted = new Date(data.submitted_at);
        const completed = data.completed_at ? new Date(data.completed_at) : new Date();
        const durationMs = completed - submitted;
        
        // Formatage du temps d'exécution
        const hours = Math.floor(durationMs / (1000 * 60 * 60));
        const minutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60));
        setExecutionTime(`${hours}h ${minutes}m`);
        
        // Estimation du temps restant basée sur la progression
        if (data.status !== 'COMPLETED' && data.tasks) {
          const progress = calculateProgress(data.tasks);
          if (progress > 0) {
            const totalEstimatedMs = (durationMs / progress) * 100;
            const remainingMs = totalEstimatedMs - durationMs;
            const remainingHours = Math.floor(remainingMs / (1000 * 60 * 60));
            const remainingMinutes = Math.floor((remainingMs % (1000 * 60 * 60)) / (1000 * 60));
            setRemainingTime(`${remainingHours}h ${remainingMinutes}m`);
          }
        }
      }
      
      // Convertir les tâches en noeuds pour ReactFlow
      if (data.tasks && Array.isArray(data.tasks)) {
        createWorkflowGraph(data.tasks);
      }
    } catch (error) {
      console.error('Erreur lors du chargement du workflow:', error);
      setError('Impossible de charger les détails du workflow. Veuillez réessayer plus tard.');
    } finally {
      setLoading(false);
    }
  };

  // Création du graphe de workflow
  const createWorkflowGraph = (tasks) => {
    // Créer les noeuds représentant les tâches
    const taskNodes = tasks.map((task, index) => ({
      id: task.id,
      type: 'special', // Type personnalisé
      position: { 
        x: 100 + (index % 3) * 250, 
        y: 100 + Math.floor(index / 3) * 150 
      },
      data: { 
        label: task.name || `Tâche ${index + 1}`, 
        status: task.status || 'PENDING', 
        progress: task.progress || 0 
      }
    }));
    
    // Créer les connexions entre noeuds basées sur les dépendances
    const taskEdges = [];
    tasks.forEach(task => {
      if (task.dependencies && Array.isArray(task.dependencies) && task.dependencies.length > 0) {
        task.dependencies.forEach(depId => {
          taskEdges.push({
            id: `e-${depId}-${task.id}`,
            source: depId,
            target: task.id,
            animated: true,
            style: { 
              stroke: '#0A2463', 
              strokeWidth: 2 
            }
          });
        });
      }
    });
    
    setNodes(taskNodes);
    setEdges(taskEdges);
  };

  // Gestionnaires d'événements
  const handleStartWorkflow = async () => {
    try {
      setLoading(true);
      await submitWorkflow(id);
      showSnackbar('Workflow démarré avec succès', 'success');
      await loadWorkflowData();
    } catch (error) {
      console.error('Erreur lors du démarrage du workflow:', error);
      showSnackbar('Erreur lors du démarrage du workflow', 'error');
      setLoading(false);
    }
  };

  const handlePauseWorkflow = async () => {
    try {
      setLoading(true);
      await pauseWorkflow(id);
      showSnackbar('Workflow mis en pause', 'info');
      await loadWorkflowData();
    } catch (error) {
      console.error('Erreur lors de la mise en pause du workflow:', error);
      showSnackbar('Erreur lors de la mise en pause du workflow', 'error');
      setLoading(false);
    }
  };

  const handleResumeWorkflow = async () => {
    try {
      setLoading(true);
      await resumeWorkflow(id);
      showSnackbar('Workflow repris avec succès', 'success');
      await loadWorkflowData();
    } catch (error) {
      console.error('Erreur lors de la reprise du workflow:', error);
      showSnackbar('Erreur lors de la reprise du workflow', 'error');
      setLoading(false);
    }
  };

  const handleRestartWorkflow = async () => {
    try {
      setLoading(true);
      // D'abord annuler le workflow existant
      await cancelWorkflow(id);
      // Puis soumettre à nouveau
      await submitWorkflow(id);
      showSnackbar('Workflow redémarré avec succès', 'success');
      await loadWorkflowData();
    } catch (error) {
      console.error('Erreur lors du redémarrage du workflow:', error);
      showSnackbar('Erreur lors du redémarrage du workflow', 'error');
      setLoading(false);
    }
  };

  const handleDeleteWorkflow = async () => {
    // Confirmation avant suppression
    if (window.confirm('Êtes-vous sûr de vouloir supprimer ce workflow ?')) {
      try {
        setLoading(true);
        // TODO: Implémenter l'API de suppression
        showSnackbar('Workflow supprimé avec succès', 'success');
        // Rediriger vers la page d'accueil
        navigate('/');
      } catch (error) {
        console.error('Erreur lors de la suppression du workflow:', error);
        showSnackbar('Erreur lors de la suppression du workflow', 'error');
        setLoading(false);
      }
    }
  };

  const handleBackToHome = () => {
    navigate('/');
  };

  // Afficher un message dans le snackbar
  const showSnackbar = (message, severity = 'info') => {
    setSnackbar({
      open: true,
      message,
      severity
    });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Affichage du chargement
  if (loading && !workflow) {
    return (
      <Backdrop open={true} sx={{ color: '#fff', zIndex: 1000 }}>
        <CircularProgress color="inherit" />
      </Backdrop>
    );
  }

  // Affichage de l'erreur
  if (error && !workflow) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button startIcon={<ArrowBack />} onClick={handleBackToHome}>
          Retour à l'accueil
        </Button>
      </Box>
    );
  }

  // Si aucune donnée n'est disponible
  if (!workflow) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="warning">
          Aucune donnée disponible pour ce workflow
        </Alert>
        <Button startIcon={<ArrowBack />} onClick={handleBackToHome} sx={{ mt: 2 }}>
          Retour à l'accueil
        </Button>
      </Box>
    );
  }

  // Calcul des statistiques
  const progress = calculateProgress(workflow.tasks);
  const taskStats = calculateTaskStats(workflow.tasks);
  const statusColor = getStatusColor(workflow.status);

  // Déterminer le bouton principal en fonction du statut
  const renderPrimaryActionButton = () => {
    switch (workflow.status) {
      case 'RUNNING':
      case 'ASSIGNING':
      case 'SPLITTING':
        return (
          <ActionButton 
            variant="contained" 
            startIcon={<Pause />}
            color="#FFC107"
            onClick={handlePauseWorkflow}
          >
            Suspendre
          </ActionButton>
        );
      case 'PAUSED':
        return (
          <ActionButton 
            variant="contained" 
            startIcon={<PlayArrow />}
            color="#4CAF50"
            onClick={handleResumeWorkflow}
          >
            Reprendre
          </ActionButton>
        );
      case 'COMPLETED':
      case 'FAILED':
        return (
          <ActionButton 
            variant="contained" 
            startIcon={<Refresh />}
            color="#4CAF50"
            onClick={handleRestartWorkflow}
          >
            Redémarrer
          </ActionButton>
        );
      default:
        return (
          <ActionButton 
            variant="contained" 
            startIcon={<PlayArrow />}
            color="#4CAF50"
            onClick={handleStartWorkflow}
          >
            Démarrer
          </ActionButton>
        );
    }
  };

  return (
    <Box
      sx={{
        minHeight: 'calc(100vh - 64px)',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%)',
        overflow: 'auto',
        p: 3
      }}
    >
      <Box sx={{ position: 'relative' }}>
        {/* Overlay de chargement */}
        {loading && (
          <Backdrop
            sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
            open={loading}
          >
            <CircularProgress color="inherit" />
          </Backdrop>
        )}
        
        {/* Bouton de retour */}
        <Button 
          startIcon={<ArrowBack />} 
          onClick={handleBackToHome}
          sx={{ mb: 2 }}
        >
          Retour à la liste
        </Button>

        {/* En-tête avec actions */}
        <GlassCard sx={{ mb: 3, p: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={7}>
              <Typography variant="h4" fontWeight="bold">{workflow.name}</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1, flexWrap: 'wrap', gap: 1 }}>
                <Chip 
                  label={workflow.status} 
                  sx={{ 
                    backgroundColor: statusColor,
                    color: '#fff',
                    fontWeight: 'bold'
                  }}
                />
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Créé le {new Date(workflow.created_at).toLocaleDateString()}
                </Typography>
                <Divider orientation="vertical" flexItem sx={{ mx: 1, height: 20 }} />
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Type: {workflow.workflow_type ? 
                    workflow.workflow_type.replace('_', ' ').toLowerCase() : 'Non spécifié'}
                </Typography>
                {workflow.owner_name && (
                  <>
                    <Divider orientation="vertical" flexItem sx={{ mx: 1, height: 20 }} />
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      Propriétaire: {workflow.owner_name}
                    </Typography>
                  </>
                )}
              </Box>
              {workflow.description && (
                <Typography variant="body2" sx={{ mt: 2, color: 'text.secondary' }}>
                  {workflow.description}
                </Typography>
              )}
            </Grid>
            <Grid item xs={12} md={5} sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, flexWrap: 'wrap' }}>
              {renderPrimaryActionButton()}
              <ActionButton 
                variant="contained" 
                startIcon={<Refresh />}
                color="#0A2463"
                onClick={handleRestartWorkflow}
                disabled={loading}
              >
                Redémarrer
              </ActionButton>
              <IconButton 
                sx={{ color: '#D90429' }} 
                onClick={handleDeleteWorkflow}
                disabled={loading}
              >
                <Delete />
              </IconButton>
            </Grid>
          </Grid>
        </GlassCard>

        {/* Statistiques */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={3}>
            <StatsCard accentColor="#4CAF50">
              <Typography variant="overline" fontWeight="bold">Progression</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <Box sx={{ flexGrow: 1, mr: 1 }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={progress} 
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
                <Typography variant="h6">{progress}%</Typography>
              </Box>
            </StatsCard>
          </Grid>
          <Grid item xs={12} md={3}>
            <StatsCard accentColor="#2196F3">
              <Typography variant="overline" fontWeight="bold">Tâches</Typography>
              <Typography variant="h4">{workflow.tasks ? workflow.tasks.length : 0}</Typography>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Terminées: {taskStats.completed}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  En cours: {taskStats.running}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  En attente: {taskStats.pending}
                </Typography>
              </Box>
            </StatsCard>
          </Grid>
          <Grid item xs={12} md={3}>
            <StatsCard accentColor="#FF9800">
              <Typography variant="overline" fontWeight="bold">Temps d'exécution</Typography>
              <Typography variant="h4">{executionTime}</Typography>
              {workflow.status !== 'COMPLETED' && (
                <Typography variant="caption" color="text.secondary">
                  Temps estimé restant: {remainingTime}
                </Typography>
              )}
            </StatsCard>
          </Grid>
          <Grid item xs={12} md={3}>
            <StatsCard accentColor="#9C27B0">
              <Typography variant="overline" fontWeight="bold">Volontaires</Typography>
              <Typography variant="h4">
                {workflow.volunteer_count || workflow.max_volunteers || '?'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Min: {workflow.min_volunteers || 1}, Max: {workflow.max_volunteers || '?'}
              </Typography>
            </StatsCard>
          </Grid>
        </Grid>

        {/* Visualisation du graphe de workflow */}
        <GlassCard sx={{ mb: 3, height: 500 }}>
          <Box
            sx={{
              display: 'flex',
              height: '100%',
              borderRadius: 2,
              overflow: 'hidden'
            }}
          >
            {nodes.length > 0 ? (
              <ReactFlow
                nodes={nodes}
                edges={edges}
                fitView
                nodeTypes={{ special: TaskNode }}
              >
                <Background color="#aaa" gap={16} />
                <Controls />
              </ReactFlow>
            ) : (
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center',
                width: '100%',
                height: '100%',
                backgroundColor: 'rgba(255, 255, 255, 0.5)',
              }}>
                <Typography variant="h6" color="text.secondary">
                  Aucune tâche disponible pour visualisation
                </Typography>
              </Box>
            )}
          </Box>
        </GlassCard>

        {/* Actions complémentaires */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <ActionButton 
              fullWidth
              variant="contained" 
              startIcon={<Assessment />}
              color="#0A2463"
              sx={{ p: 2, justifyContent: 'flex-start' }}
              onClick={() => navigate(`/results/${workflow.id}`)}
            >
              Visualiser les résultats
            </ActionButton>
          </Grid>
          <Grid item xs={12} md={4}>
            <ActionButton 
              fullWidth
              variant="contained" 
              startIcon={<Storage />}
              color="#4CAF50"
              sx={{ p: 2, justifyContent: 'flex-start' }}
            >
              Explorer les données
            </ActionButton>
          </Grid>
          <Grid item xs={12} md={4}>
            <ActionButton 
              fullWidth
              variant="contained" 
              startIcon={<CloudDownload />}
              color="#9C27B0"
              sx={{ p: 2, justifyContent: 'flex-start' }}
            >
              Télécharger le rapport
            </ActionButton>
          </Grid>
        </Grid>
      </Box>

      {/* Snackbar pour les notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default WorkflowDetail;