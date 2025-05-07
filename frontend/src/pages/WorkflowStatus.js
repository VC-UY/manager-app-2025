import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Box, Grid, Typography, Card, Divider, Button, Chip,
  IconButton, LinearProgress, Backdrop, CircularProgress,
  Alert, Snackbar, Paper, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Tabs, Tab, Accordion,
  AccordionSummary, AccordionDetails, Tooltip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { 
  PlayArrow, Pause, Refresh, ArrowBack, ExpandMore,
  Visibility, CloudDownload, Assessment, Delete, 
  CheckCircle, Cancel, HourglassEmpty, MemoryOutlined,
  Speed, Storage, CloudQueue, Group
} from '@mui/icons-material';
import { 
  fetchWorkflowDetails, 
  submitWorkflow, 
  pauseWorkflow, 
  resumeWorkflow, 
  cancelWorkflow,
  fetchWorkflowTasks,
  fetchWorkflowResultInfo
} from '../services/api';
import coordinationService from '../services/CoordinationService';

// Composants stylisés
const StatusCard = styled(Card)(({ theme }) => ({
  borderRadius: 16,
  padding: theme.spacing(3),
  boxShadow: '0 8px 32px rgba(10, 36, 99, 0.15)',
  backdropFilter: 'blur(10px)',
  background: 'rgba(14, 28, 54, 0.7)',
  transition: 'all 0.3s ease',
  border: '1px solid rgba(255, 255, 255, 0.1)',
}));

const ActionButton = styled(Button)(({ theme, color }) => ({
  borderRadius: 12,
  padding: '10px 16px',
  background: color || theme.palette.primary.main,
  color: 'white',
  '&:hover': {
    background: color ? `${color}dd` : theme.palette.primary.dark,
    boxShadow: '0 6px 16px rgba(0, 0, 0, 0.2)',
    transform: 'translateY(-2px)',
  },
  transition: 'all 0.3s ease',
}));

const StatsCard = styled(Box)(({ theme, accentColor }) => ({
  background: 'rgba(255, 255, 255, 0.05)',
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

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
  color: theme.palette.text.primary,
}));

const StatusChip = styled(Chip)(({ status }) => {
  const getColor = () => {
    switch (status) {
      case 'RUNNING': return '#4CAF50';
      case 'COMPLETED': return '#2196F3';
      case 'FAILED': return '#D90429';
      case 'PAUSED': return '#FFC107';
      case 'CREATED': return '#9E9E9E';
      case 'VALIDATED': return '#9C27B0';
      case 'SUBMITTED': return '#03A9F4';
      case 'SPLITTING': return '#00BCD4';
      case 'ASSIGNING': return '#009688';
      case 'PENDING': return '#FF9800';
      case 'PARTIAL_FAILURE': return '#FF5722';
      case 'REASSIGNING': return '#795548';
      case 'AGGREGATING': return '#607D8B';
      default: return '#9E9E9E';
    }
  };
  
  return {
    backgroundColor: getColor(),
    color: 'white',
    fontWeight: 'bold',
    '& .MuiChip-icon': {
      color: 'white',
    }
  };
});

// Fonction utilitaire pour formater le temps
const formatDuration = (milliseconds) => {
  if (!milliseconds) return '--';
  
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  
  return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
};

// Fonction pour calculer la progression
const calculateProgress = (tasks) => {
  if (!tasks || tasks.length === 0) return 0;
  
  const completedTasks = tasks.filter(task => task.status === 'COMPLETED').length;
  return Math.round((completedTasks / tasks.length) * 100);
};

// Composant principal
const WorkflowStatus = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [workflow, setWorkflow] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  const [executionTime, setExecutionTime] = useState('--');
  const [estimatedTime, setEstimatedTime] = useState('--');
  
  // Fonction pour charger les détails du workflow
  const loadWorkflowDetails = useCallback(async () => {
    try {
      setRefreshing(true);
      
      // Charger les détails du workflow
      const workflowData = await fetchWorkflowDetails(id);
      setWorkflow(workflowData);
      
      // Calculer le temps d'exécution
      if (workflowData.submitted_at) {
        const submittedDate = new Date(workflowData.submitted_at);
        const endDate = workflowData.completed_at ? new Date(workflowData.completed_at) : new Date();
        const duration = endDate - submittedDate;
        setExecutionTime(formatDuration(duration));
        
        // Calculer le temps estimé restant
        if (workflowData.status !== 'COMPLETED') {
          try {
            // Charger les tâches pour calculer la progression
            const tasksData = await fetchWorkflowTasks(id);
            if (Array.isArray(tasksData)) {
              setTasks(tasksData);
              const progress = calculateProgress(tasksData);
              
              if (progress > 0) {
                const totalEstimatedTime = (duration / progress) * 100;
                const remainingTime = totalEstimatedTime - duration;
                setEstimatedTime(formatDuration(remainingTime));
              }
            }
          } catch (error) {
            console.warn('Erreur lors du calcul du temps estimé:', error);
          }
        }
      }
      
      // Charger les informations sur les résultats si le workflow est terminé
      if (workflowData.status === 'COMPLETED') {
        try {
          const resultsData = await fetchWorkflowResultInfo(id);
          setResults(resultsData);
        } catch (error) {
          console.warn('Erreur lors du chargement des résultats:', error);
        }
      }
    } catch (error) {
      console.error('Erreur lors du chargement des détails du workflow:', error);
      setError('Impossible de charger les détails du workflow. Veuillez réessayer plus tard.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [id]);

  // Chargement des données au montage
  useEffect(() => {
    loadWorkflowDetails();
    
    // S'abonner aux mises à jour du statut du workflow
    const unsubscribe = coordinationService.subscribeToWorkflowStatus(id, (status) => {
      if (status && status.status) {
        loadWorkflowDetails();
      }
    });
    
    // Mise à jour automatique toutes les 10 secondes si le workflow est en cours
    const interval = setInterval(() => {
      if (workflow && ['RUNNING', 'SPLITTING', 'ASSIGNING', 'PENDING'].includes(workflow.status)) {
        loadWorkflowDetails();
      }
    }, 10000);
    
    // Nettoyage à la déconnexion
    return () => {
      clearInterval(interval);
      if (unsubscribe) unsubscribe();
    };
  }, [id, loadWorkflowDetails, workflow]);
  
  // Gestionnaires d'événements
  const handleRefresh = () => {
    loadWorkflowDetails();
  };
  
  const handleStartWorkflow = async () => {
    try {
      setRefreshing(true);
      await submitWorkflow(id);
      showSnackbar('Workflow démarré avec succès', 'success');
      await loadWorkflowDetails();
    } catch (error) {
      console.error('Erreur lors du démarrage du workflow:', error);
      showSnackbar('Erreur lors du démarrage du workflow', 'error');
      setRefreshing(false);
    }
  };
  
  const handlePauseWorkflow = async () => {
    try {
      setRefreshing(true);
      await pauseWorkflow(id);
      showSnackbar('Workflow mis en pause', 'info');
      await loadWorkflowDetails();
    } catch (error) {
      console.error('Erreur lors de la mise en pause du workflow:', error);
      showSnackbar('Erreur lors de la mise en pause du workflow', 'error');
      setRefreshing(false);
    }
  };
  
  const handleResumeWorkflow = async () => {
    try {
      setRefreshing(true);
      await resumeWorkflow(id);
      showSnackbar('Workflow repris avec succès', 'success');
      await loadWorkflowDetails();
    } catch (error) {
      console.error('Erreur lors de la reprise du workflow:', error);
      showSnackbar('Erreur lors de la reprise du workflow', 'error');
      setRefreshing(false);
    }
  };
  
  const handleCancelWorkflow = async () => {
    // Confirmation avant annulation
    if (window.confirm('Êtes-vous sûr de vouloir annuler ce workflow ?')) {
      try {
        setRefreshing(true);
        await cancelWorkflow(id);
        showSnackbar('Workflow annulé', 'info');
        await loadWorkflowDetails();
      } catch (error) {
        console.error('Erreur lors de l\'annulation du workflow:', error);
        showSnackbar('Erreur lors de l\'annulation du workflow', 'error');
        setRefreshing(false);
      }
    }
  };
  
  const handleDeleteWorkflow = () => {
    // Confirmation avant suppression
    if (window.confirm('Êtes-vous sûr de vouloir supprimer ce workflow ? Cette action est irréversible.')) {
      showSnackbar('Fonctionnalité de suppression non implémentée', 'warning');
    }
  };
  
  const handleViewResults = () => {
    navigate(`/results/${id}`);
  };
  
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  const handleDownloadResults = () => {
    if (!workflow || workflow.status !== 'COMPLETED') {
      showSnackbar('Les résultats ne sont pas encore disponibles', 'warning');
      return;
    }
    
    showSnackbar('Téléchargement des résultats en cours...', 'info');
    // Logique de téléchargement à implémenter
  };
  
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
  
  const handleBackToHome = () => {
    navigate('/');
  };

  // Affichage pendant le chargement
  if (loading && !workflow) {
    return (
      <Backdrop open={true} sx={{ color: '#fff', zIndex: 1000 }}>
        <CircularProgress color="inherit" />
      </Backdrop>
    );
  }

  // Affichage en cas d'erreur
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

  // Si pas de données workflow
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

  // Déterminer le bouton d'action principal
  const renderActionButton = () => {
    switch (workflow.status) {
      case 'RUNNING':
      case 'SPLITTING':
      case 'ASSIGNING':
      case 'PENDING':
        return (
          <ActionButton 
            variant="contained" 
            startIcon={<Pause />}
            color="#FFC107"
            onClick={handlePauseWorkflow}
            disabled={refreshing}
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
            disabled={refreshing}
          >
            Reprendre
          </ActionButton>
        );
      case 'COMPLETED':
      case 'FAILED':
      case 'PARTIAL_FAILURE':
        return (
          <ActionButton 
            variant="contained" 
            startIcon={<Refresh />}
            color="#2196F3"
            onClick={handleStartWorkflow}
            disabled={refreshing}
          >
            Réexécuter
          </ActionButton>
        );
      default:
        return (
          <ActionButton 
            variant="contained" 
            startIcon={<PlayArrow />}
            color="#4CAF50"
            onClick={handleStartWorkflow}
            disabled={refreshing}
          >
            Démarrer
          </ActionButton>
        );
    }
  };

  // Calculer les statistiques de tâches
  const taskStats = {
    total: tasks.length,
    completed: tasks.filter(task => task.status === 'COMPLETED').length,
    running: tasks.filter(task => ['RUNNING', 'ASSIGNING'].includes(task.status)).length,
    pending: tasks.filter(task => ['PENDING', 'CREATED'].includes(task.status)).length,
    failed: tasks.filter(task => task.status === 'FAILED').length
  };

  // Calculer le pourcentage de progression
  const progress = calculateProgress(tasks);

  return (
    <Box
      sx={{
        minHeight: 'calc(100vh - 64px)',
        overflow: 'auto',
        p: 3
      }}
    >
      {/* Overlay de chargement */}
      {refreshing && (
        <Backdrop
          sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
          open={refreshing}
        >
          <CircularProgress color="inherit" />
        </Backdrop>
      )}

      {/* Bouton de retour */}
      <Button 
        startIcon={<ArrowBack />} 
        variant="outlined"
        onClick={handleBackToHome}
        sx={{ mb: 2 }}
      >
        Retour à la liste
      </Button>

     {/* En-tête avec informations principales */}
    <StatusCard sx={{ mb: 3 }}>
      <Grid container spacing={2}>
        <Grid item xs={12} md={7}>
          <Typography variant="h4" fontWeight="bold" sx={{ color: 'white' }}>
            {workflow.name}
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ color: 'white' }}>
              Tâches du Workflow ({tasks.length})
            </Typography>
            <Button 
              startIcon={<Refresh />} 
              variant="outlined" 
              onClick={handleRefresh}
              disabled={refreshing}
            >
              Actualiser
            </Button>
          </Box>
          
          {tasks.length > 0 ? (
            <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <StyledTableCell>ID</StyledTableCell>
                    <StyledTableCell>Nom</StyledTableCell>
                    <StyledTableCell>Statut</StyledTableCell>
                    <StyledTableCell>Progression</StyledTableCell>
                    <StyledTableCell>Créée le</StyledTableCell>
                    <StyledTableCell>Actions</StyledTableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {tasks.map((task) => (
                    <TableRow key={task.id}>
                      <StyledTableCell>{task.id ? task.id.substring(0, 8) + '...' : 'N/A'}</StyledTableCell>
                      <StyledTableCell>{task.name || `Tâche ${task.id ? task.id.substring(0, 6) : ''}`}</StyledTableCell>
                      <StyledTableCell>
                        <StatusChip 
                          label={task.status || 'UNKNOWN'} 
                          status={task.status || 'UNKNOWN'}
                          size="small"
                        />
                      </StyledTableCell>
                      <StyledTableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <LinearProgress 
                            variant="determinate" 
                            value={task.progress || 0} 
                            sx={{ 
                              width: 100, 
                              mr: 1, 
                              height: 8, 
                              borderRadius: 4,
                              backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            }}
                          />
                          <Typography variant="body2">{task.progress || 0}%</Typography>
                        </Box>
                      </StyledTableCell>
                      <StyledTableCell>
                        {task.created_at ? new Date(task.created_at).toLocaleString() : 'N/A'}
                      </StyledTableCell>
                      <StyledTableCell>
                        <Tooltip title="Voir les détails">
                          <IconButton 
                            size="small" 
                            onClick={() => console.log('Voir détails de la tâche', task.id)}
                            sx={{ color: 'white' }}
                          >
                            <Visibility />
                          </IconButton>
                        </Tooltip>
                      </StyledTableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">
              Aucune tâche disponible pour ce workflow.
            </Alert>
          )}
        </Grid>
      </Grid>
    </StatusCard>

    {/* Onglet Paramètres */}
    {tabValue === 2 && (
      <StatusCard>
        <Typography variant="h6" sx={{ mb: 3, color: 'white' }}>
          Paramètres du Workflow
        </Typography>
        
        {workflow.metadata && Object.keys(workflow.metadata).length > 0 ? (
          <Box>
            {/* Paramètres généraux */}
            <Accordion 
              defaultExpanded 
              sx={{ 
                mb: 2, 
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                color: 'white',
                '&:before': {
                  display: 'none',
                },
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMore sx={{ color: 'white' }} />}
              >
                <Typography>Paramètres généraux</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      Temps d'exécution maximum
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                      {workflow.max_execution_time ? formatDuration(workflow.max_execution_time * 1000) : 'Non spécifié'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      Tentatives en cas d'échec
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                      {workflow.retry_count || 'Non spécifié'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      Priorité
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                      {workflow.priority || 'Normal (1)'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      Propriétaire
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                      {workflow.owner_name || 'Non spécifié'}
                    </Typography>
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* Paramètres des volontaires */}
            <Accordion 
              sx={{ 
                mb: 2, 
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                color: 'white',
                '&:before': {
                  display: 'none',
                },
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMore sx={{ color: 'white' }} />}
              >
                <Typography>Configuration des volontaires</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      Minimum requis
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                      {workflow.min_volunteers || 'Non spécifié'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      Maximum autorisé
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                      {workflow.max_volunteers || 'Non spécifié'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      Préférences de types
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                      {workflow.volunteer_preferences && workflow.volunteer_preferences.length > 0 ? (
                        workflow.volunteer_preferences.map((pref, index) => (
                          <Chip 
                            key={index} 
                            label={pref} 
                            size="small"
                            sx={{ backgroundColor: 'rgba(76, 175, 80, 0.2)', color: 'white' }}
                          />
                        ))
                      ) : (
                        <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.5)' }}>
                          Tous types de volontaires acceptés
                        </Typography>
                      )}
                    </Box>
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* Paramètres spécifiques au type de workflow */}
            {workflow.metadata.input && (
              <Accordion 
                sx={{ 
                  mb: 2, 
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  color: 'white',
                  '&:before': {
                    display: 'none',
                  },
                }}
              >
                <AccordionSummary
                  expandIcon={<ExpandMore sx={{ color: 'white' }} />}
                >
                  <Typography>Paramètres spécifiques</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {workflow.workflow_type === 'MATRIX_ADDITION' || workflow.workflow_type === 'MATRIX_MULTIPLICATION' ? (
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Type d'opération
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.workflow_type === 'MATRIX_ADDITION' ? 'Addition de matrices' : 'Multiplication de matrices'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Algorithme
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.metadata.input.algorithm || 'Standard'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Précision
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.metadata.input.precision || 'Double'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Taille des blocs
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.metadata.input.block_size || 'Non spécifié'}
                        </Typography>
                      </Grid>
                      
                      {/* Matrice A */}
                      <Grid item xs={12}>
                        <Typography variant="subtitle1" sx={{ color: 'white', mt: 2 }}>
                          Matrice A
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Format
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.metadata.input.matrix_a?.format || 'Non spécifié'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Dimensions
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.metadata.input.matrix_a?.dimensions 
                            ? `${workflow.metadata.input.matrix_a.dimensions[0]} × ${workflow.metadata.input.matrix_a.dimensions[1]}`
                            : 'Non spécifié'
                          }
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Stockage
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.metadata.input.matrix_a?.storage_type || 'Non spécifié'}
                        </Typography>
                      </Grid>
                      
                      {/* Matrice B */}
                      <Grid item xs={12}>
                        <Typography variant="subtitle1" sx={{ color: 'white', mt: 2 }}>
                          Matrice B   
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Format
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.metadata.input.matrix_b?.format || 'Non spécifié'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Dimensions
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.metadata.input.matrix_b?.dimensions 
                            ? `${workflow.metadata.input.matrix_b.dimensions[0]} × ${workflow.metadata.input.matrix_b.dimensions[1]}`
                            : 'Non spécifié'
                          }
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          Stockage
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {workflow.metadata.input.matrix_b?.storage_type || 'Non spécifié'}
                        </Typography>
                      </Grid>
                    </Grid>
                  ) : (
                    <Box sx={{ p: 2 }}>
                      <pre style={{ color: 'white', overflow: 'auto' }}>
                        {JSON.stringify(workflow.metadata.input, null, 2)}
                      </pre>
                    </Box>
                  )}
                </AccordionDetails>
              </Accordion>
            )}

            {/* Ressources estimées */}
            {workflow.estimated_resources && Object.keys(workflow.estimated_resources).length > 0 && (
              <Accordion 
                sx={{ 
                  mb: 2, 
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  color: 'white',
                  '&:before': {
                    display: 'none',
                  },
                }}
              >
                <AccordionSummary
                  expandIcon={<ExpandMore sx={{ color: 'white' }} />}
                >
                  <Typography>Ressources estimées</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    {Object.entries(workflow.estimated_resources).map(([key, value]) => (
                      <Grid item xs={12} sm={6} key={key}>
                        <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          {key.charAt(0).toUpperCase() + key.slice(1).replace('_', ' ')}
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2, color: 'white' }}>
                          {value}
                        </Typography>
                      </Grid>
                    ))}
                  </Grid>
                </AccordionDetails>
              </Accordion>
            )}
          </Box>
        ) : (
          <Alert severity="info">
            Aucun paramètre spécifique disponible pour ce workflow.
          </Alert>
        )}
      </StatusCard>
    )}

    {/* Onglet Résultats (affiché uniquement si le workflow est terminé) */}
    {tabValue === 3 && workflow.status === 'COMPLETED' && (
      <StatusCard>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6" sx={{ color: 'white' }}>
            Résultats du Workflow
          </Typography>
          <Button
            variant="contained"
            startIcon={<CloudDownload />}
            onClick={handleDownloadResults}
            disabled={!results}
            color="secondary"
          >
            Télécharger les résultats
          </Button>
        </Box>
        
        {results ? (
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 2, color: 'white' }}>
              Informations sur les résultats
            </Typography>
            
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={4}>
                <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Format des résultats
                </Typography>
                <Typography variant="body1" sx={{ color: 'white' }}>
                  {results.format || 'Non spécifié'}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Taille
                </Typography>
                <Typography variant="body1" sx={{ color: 'white' }}>
                  {results.size ? `${(results.size / 1024).toFixed(2)} KB` : 'Non spécifié'}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Généré le
                </Typography>
                <Typography variant="body1" sx={{ color: 'white' }}>
                  {results.generated_at ? new Date(results.generated_at).toLocaleString() : 'Non spécifié'}
                </Typography>
              </Grid>
            </Grid>
            
            <Button 
              variant="outlined" 
              color="secondary"
              startIcon={<Assessment />}
              onClick={handleViewResults}
              sx={{ mt: 2 }}
            >
              Visualiser les résultats
            </Button>
          </Box>
        ) : (
          <Alert severity="info">
            Les informations sur les résultats ne sont pas disponibles. Essayez de rafraîchir la page.
          </Alert>
        )}
      </StatusCard>
    )}

      {/* Actions complémentaires en bas de page */}
      <Grid container spacing={3} sx={{ mt: 3 }}>
        <Grid item xs={12} md={4}>
          <ActionButton 
            fullWidth
            variant="contained" 
            startIcon={<Assessment />}
            color="#0A2463"
            sx={{ p: 2, justifyContent: 'flex-start' }}
            onClick={handleViewResults}
            disabled={workflow.status !== 'COMPLETED'}
          >
            Visualiser les résultats
          </ActionButton>
        </Grid>
        <Grid item xs={12} md={4}>
          <ActionButton 
            fullWidth
            variant="contained" 
            startIcon={<CloudDownload />}
            color="#4CAF50"
            sx={{ p: 2, justifyContent: 'flex-start' }}
            onClick={handleDownloadResults}
            disabled={workflow.status !== 'COMPLETED'}
          >
            Télécharger les résultats
          </ActionButton>
        </Grid>
        <Grid item xs={12} md={4}>
          <ActionButton 
            fullWidth
            variant="contained" 
            startIcon={workflow.status === 'RUNNING' ? <Pause /> : <PlayArrow />}
            color={workflow.status === 'RUNNING' ? "#FFC107" : "#4CAF50"}
            sx={{ p: 2, justifyContent: 'flex-start' }}
            onClick={workflow.status === 'RUNNING' ? handlePauseWorkflow : handleStartWorkflow}
            disabled={['COMPLETED', 'FAILED'].includes(workflow.status) || refreshing}
          >
            {workflow.status === 'RUNNING' ? 'Suspendre le workflow' : 'Exécuter le workflow'}
          </ActionButton>
        </Grid>
      </Grid>

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

// Fonction utilitaire pour obtenir la couleur de progression en fonction du statut
const getProgressColor = (status) => {
  switch (status) {
    case 'RUNNING':
    case 'SPLITTING':
    case 'ASSIGNING': 
      return '#4CAF50'; // Vert
    case 'PAUSED':
      return '#FFC107'; // Jaune
    case 'COMPLETED':
      return '#2196F3'; // Bleu
    case 'FAILED':
    case 'PARTIAL_FAILURE':
      return '#D90429'; // Rouge
    default:
      return '#9E9E9E'; // Gris
  }
};

// Fonction pour obtenir une description du statut
const getStatusDescription = (status) => {
  switch (status) {
    case 'CREATED':
      return 'Le workflow a été créé mais n\'a pas encore été validé ou soumis.';
    case 'VALIDATED':
      return 'Le workflow a été validé et est prêt à être soumis.';
    case 'SUBMITTED':
      return 'Le workflow a été soumis et est en attente de traitement.';
    case 'SPLITTING':
      return 'Les tâches du workflow sont en cours de découpage.';
    case 'ASSIGNING':
      return 'Les tâches sont en cours d\'attribution aux volontaires.';
    case 'PENDING':
      return 'Le workflow est en attente d\'exécution par les volontaires.';
    case 'RUNNING':
      return 'Le workflow est en cours d\'exécution.';
    case 'PAUSED':
      return 'L\'exécution du workflow a été temporairement suspendue.';
    case 'PARTIAL_FAILURE':
      return 'Certaines tâches du workflow ont échoué, mais d\'autres continuent.';
    case 'REASSIGNING':
      return 'Certaines tâches sont en cours de réattribution après échec.';
    case 'AGGREGATING':
      return 'Les résultats des tâches sont en cours d\'agrégation.';
    case 'COMPLETED':
      return 'Le workflow s\'est terminé avec succès.';
    case 'FAILED':
      return 'Le workflow a échoué et ne peut pas être complété.';
    default:
      return 'Statut inconnu.';
  }
};

export default WorkflowStatus;