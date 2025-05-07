import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Card, Grid, Button, Chip, 
  CircularProgress, Table, TableBody, TableCell, 
  TableContainer, TableHead, TableRow, Paper, TextField,
  InputAdornment, Alert, Snackbar
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';
import { Add, Search, Visibility, PlayArrow, Pause, Refresh } from '@mui/icons-material';
import api, { fetchWorkflows } from '../services/api';

// Composants stylisés
const GlowingCard = styled(Card)(({ theme }) => ({
  padding: theme.spacing(3),
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: '0 0 20px rgba(76, 175, 80, 0.3)',
    transform: 'translateY(-5px)',
  }
}));

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  backgroundColor: 'rgba(10, 36, 99, 0.4)',
  color: theme.palette.text.primary,
  borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
  '&:nth-of-type(odd)': {
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
  },
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  '& > *': {
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
  },
}));

const StatusChip = styled(Chip)(({ status }) => {
  const getColor = () => {
    switch (status) {
      case 'RUNNING': return '#4CAF50';
      case 'COMPLETED': return '#2196F3';
      case 'FAILED': return '#D90429';
      case 'PENDING': return '#FF9800';
      case 'PAUSED': return '#9C27B0';
      case 'SUBMITTED': return '#03A9F4';
      case 'SPLITTING': return '#00BCD4';
      case 'ASSIGNING': return '#009688';
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

// Composant principal
const Home = () => {
  // États
  const [workflows, setWorkflows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('info');
  
  const navigate = useNavigate();

  // Chargement initial des workflows
  useEffect(() => {
    loadWorkflows();
  }, []);

  // Fonction pour charger les workflows
  const loadWorkflows = async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('Chargement des workflows...');
      const data = await fetchWorkflows();
      console.log('Données reçues:', data);
      
      // S'assurer que data est un tableau
      let workflowsArray = [];
      
      if (Array.isArray(data)) {
        workflowsArray = data;
      } else if (data && typeof data === 'object') {
        if (Array.isArray(data.results)) {
          workflowsArray = data.results;
        } else if (data.workflows && Array.isArray(data.workflows)) {
          workflowsArray = data.workflows;
        }
      }
      
      console.log('Workflows formatés:', workflowsArray);
      setWorkflows(workflowsArray);
      
      // Afficher un message si aucun workflow n'est trouvé
      if (workflowsArray.length === 0) {
        showSnackbar('Aucun workflow trouvé. Créez-en un nouveau pour commencer.', 'info');
      }
    } catch (error) {
      console.error('Erreur lors du chargement des workflows:', error);
      setError('Impossible de charger les workflows. Veuillez réessayer plus tard.');
      setWorkflows([]);
    } finally {
      setLoading(false);
    }
  };

  // Fonction utilitaire pour afficher des messages toast
  const showSnackbar = (message, severity = 'info') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };

  // Filtrer les workflows selon le terme de recherche
  const filteredWorkflows = React.useMemo(() => {
    if (!Array.isArray(workflows)) return [];
    
    return workflows.filter(workflow => {
      // Vérifier que le workflow est bien défini
      if (!workflow) return false;
      
      // Chercher dans le nom (si défini)
      const nameMatch = workflow.name && 
        workflow.name.toLowerCase().includes(searchTerm.toLowerCase());
      
      // Chercher dans la description (si définie)
      const descMatch = workflow.description && 
        workflow.description.toLowerCase().includes(searchTerm.toLowerCase());
      
      // Chercher dans le type (si défini)
      const typeMatch = workflow.workflow_type && 
        workflow.workflow_type.toLowerCase().includes(searchTerm.toLowerCase());
      
      return nameMatch || descMatch || typeMatch;
    });
  }, [workflows, searchTerm]);

  // Handlers pour les actions
  const handleCreateWorkflow = () => {
    navigate('/create');
  };

  const handleExecuteWorkflow = async (id) => {
    try {
      setLoading(true);
      await api.post(`/workflows/${id}/submit/`);
      showSnackbar('Workflow soumis avec succès', 'success');
      await loadWorkflows(); // Recharger les workflows
    } catch (error) {
      console.error(`Erreur lors de l'exécution du workflow ${id}:`, error);
      showSnackbar(`Erreur lors de l'exécution du workflow: ${error.message || 'Erreur inconnue'}`, 'error');
      setLoading(false);
    }
  };

  const handlePauseWorkflow = async (id) => {
    try {
      setLoading(true);
      await api.post(`/workflows/${id}/pause/`);
      showSnackbar('Workflow mis en pause avec succès', 'success');
      await loadWorkflows(); // Recharger les workflows
    } catch (error) {
      console.error(`Erreur lors de la mise en pause du workflow ${id}:`, error);
      showSnackbar(`Erreur lors de la mise en pause du workflow: ${error.message || 'Erreur inconnue'}`, 'error');
      setLoading(false);
    }
  };

  const formatWorkflowType = (type) => {
    if (!type) return '-';
    
    // Remplacer les underscores par des espaces et formater le texte
    const formattedType = type.replace(/_/g, ' ').toLowerCase();
    
    // Mettre en majuscule la première lettre de chaque mot
    return formattedType
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Rendu du composant
  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      {/* En-tête */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mb: 4,
        px: 2,
        flexWrap: 'wrap',
        gap: 2
      }}>
        <Typography 
          variant="h3" 
          sx={{ 
            fontWeight: 'bold',
            background: 'linear-gradient(45deg, #FFFFFF 30%, #4CAF50 90%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Vos Workflows
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            color="secondary"
            startIcon={<Refresh />}
            onClick={loadWorkflows}
            disabled={loading}
            sx={{ borderRadius: '50px' }}
          >
            Actualiser
          </Button>
          <Button
            variant="contained"
            color="secondary"
            startIcon={<Add />}
            onClick={handleCreateWorkflow}
            sx={{ 
              borderRadius: '50px', 
              px: 3,
              boxShadow: '0 0 15px rgba(76, 175, 80, 0.5)',
            }}
          >
            Nouveau Workflow
          </Button>
        </Box>
      </Box>

      {/* Barre de recherche */}
      <GlowingCard sx={{ mb: 4 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Rechercher un workflow
        </Typography>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Rechercher par nom, description ou type..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search sx={{ color: 'rgba(255, 255, 255, 0.7)' }} />
              </InputAdornment>
            ),
            sx: {
              borderRadius: 2,
              backgroundColor: 'rgba(255, 255, 255, 0.05)',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
              },
            },
          }}
        />
      </GlowingCard>

      {/* Message d'erreur */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Affichage du chargement ou du tableau */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 5 }}>
          <CircularProgress color="secondary" />
        </Box>
      ) : (
        <TableContainer component={Paper} sx={{ backgroundColor: 'rgba(14, 28, 54, 0.6)' }}>
          <Table>
            <TableHead>
              <TableRow>
                <StyledTableCell>Nom</StyledTableCell>
                <StyledTableCell>Type</StyledTableCell>
                <StyledTableCell>Statut</StyledTableCell>
                <StyledTableCell>Créé le</StyledTableCell>
                <StyledTableCell>Priorité</StyledTableCell>
                <StyledTableCell>Actions</StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredWorkflows.length > 0 ? (
                filteredWorkflows.map((workflow) => (
                  <StyledTableRow key={workflow.id}>
                    <TableCell>{workflow.name || '-'}</TableCell>
                    <TableCell>{formatWorkflowType(workflow.workflow_type)}</TableCell>
                    <TableCell>
                      <StatusChip 
                        label={workflow.status || 'UNKNOWN'}
                        status={workflow.status || 'UNKNOWN'}
                        icon={workflow.status === 'RUNNING' ? <PlayArrow /> : null}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {workflow.created_at 
                        ? new Date(workflow.created_at).toLocaleDateString('fr-FR', {
                            day: 'numeric',
                            month: 'short',
                            year: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })
                        : '-'
                      }
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {workflow.priority && workflow.priority > 0 
                          ? Array.from({ length: Math.min(workflow.priority, 5) }).map((_, i) => (
                              <Box 
                                key={i}
                                sx={{ 
                                  width: 12, 
                                  height: 12, 
                                  borderRadius: '50%',
                                  backgroundColor: '#4CAF50',
                                  mr: 0.5,
                                  boxShadow: '0 0 5px #4CAF50'
                                }} 
                              />
                            ))
                          : '-'
                        }
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={<Visibility />}
                          onClick={() => navigate(`/status/${workflow.id}`)}
                          sx={{ 
                            background: 'rgba(255, 255, 255, 0.1)',
                            '&:hover': {
                              background: 'rgba(255, 255, 255, 0.2)',
                            }
                          }}
                        >
                          Détails
                        </Button>
                        {workflow.status === 'RUNNING' ? (
                          <Button
                            variant="contained"
                            size="small"
                            color="warning"
                            startIcon={<Pause />}
                            onClick={() => handlePauseWorkflow(workflow.id)}
                            sx={{ 
                              background: 'rgba(255, 152, 0, 0.5)',
                              '&:hover': {
                                background: 'rgba(255, 152, 0, 0.7)',
                              }
                            }}
                          >
                            Pause
                          </Button>
                        ) : (
                          <Button
                            variant="contained"
                            size="small"
                            color="secondary"
                            startIcon={<PlayArrow />}
                            onClick={() => handleExecuteWorkflow(workflow.id)}
                            disabled={['COMPLETED', 'FAILED'].includes(workflow.status)}
                            sx={{ 
                              background: 'rgba(76, 175, 80, 0.5)',
                              '&:hover': {
                                background: 'rgba(76, 175, 80, 0.7)',
                              }
                            }}
                          >
                            Exécuter
                          </Button>
                        )}
                      </Box>
                    </TableCell>
                  </StyledTableRow>
                ))
              ) : (
                <StyledTableRow>
                  <TableCell colSpan={6} align="center">
                    <Typography variant="body1" sx={{ py: 3 }}>
                      Aucun workflow trouvé. Créez un nouveau workflow pour commencer.
                    </Typography>
                    <Button
                      variant="contained"
                      color="secondary"
                      startIcon={<Add />}
                      onClick={handleCreateWorkflow}
                    >
                      Nouveau Workflow
                    </Button>
                  </TableCell>
                </StyledTableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Snackbar pour les notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setSnackbarOpen(false)} 
          severity={snackbarSeverity}
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Home;