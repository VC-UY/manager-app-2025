import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  Box, Typography, TextField, Grid, Button, IconButton,
  Card, CardContent, Stepper, Step, StepLabel, Divider,
  Paper, Chip, Menu, MenuItem, Tooltip, Tab, Tabs, Switch,
  FormControlLabel, Slider, Select, InputLabel, FormControl,
  CircularProgress, Alert, Snackbar, Backdrop, Autocomplete
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Save, PlayArrow, Add, Delete, ArrowBack, Settings,
  DragIndicator, Code, Visibility, CloudUpload, Link,
  Refresh, CheckCircle, Error as ErrorIcon
} from '@mui/icons-material';
import ReactFlow, {
  Background, Controls, MiniMap, addEdge, 
  useNodesState, useEdgesState
} from 'reactflow';
import 'reactflow/dist/style.css';

import {
  fetchWorkflowDetails,
  createMatrixWorkflow,
  submitWorkflow
} from '../services/api';

// Nœud de tâche personnalisé
const CustomTaskNode = ({ data, id, selected }) => {
  return (
    <Paper
      elevation={3}
      sx={{
        width: 220,
        p: 2,
        borderRadius: 3,
        background: selected 
          ? 'linear-gradient(145deg, #f0f7ff, #e6f0ff)' 
          : 'linear-gradient(145deg, #ffffff, #f5f5f5)',
        boxShadow: selected 
          ? '0 0 0 2px #2196F3, 0 10px 20px rgba(0,0,0,0.12)'
          : '0 10px 20px rgba(0,0,0,0.12)',
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: '0 15px 30px rgba(0,0,0,0.15)',
        }
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle1" fontWeight="bold">{data.label}</Typography>
        <DragIndicator sx={{ cursor: 'grab', color: 'text.secondary' }} />
      </Box>
      
      <Divider sx={{ my: 1 }} />
      
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <Typography variant="caption" sx={{ mr: 1 }}>Type:</Typography>
        <Chip 
          label={data.type || "Traitement"} 
          size="small" 
          sx={{ 
            backgroundColor: data.color || '#0A2463',
            color: '#fff',
            fontSize: '0.7rem'
          }} 
        />
      </Box>
      
      {data.parameters && (
        <Box sx={{ mt: 1, mb: 1 }}>
          <Typography variant="caption" color="text.secondary">
            {Object.keys(data.parameters).length} paramètre(s)
          </Typography>
        </Box>
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
        <Tooltip title="Paramètres">
          <IconButton size="small" onClick={data.onSettings ? () => data.onSettings(id) : undefined}>
            <Settings fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Code">
          <IconButton size="small" onClick={data.onViewCode ? () => data.onViewCode(id) : undefined}>
            <Code fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Aperçu">
          <IconButton size="small" onClick={data.onPreview ? () => data.onPreview(id) : undefined}>
            <Visibility fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Supprimer">
          <IconButton 
            size="small" 
            sx={{ color: '#D90429' }}
            onClick={data.onDelete ? () => data.onDelete(id) : undefined}
          >
            <Delete fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </Paper>
  );
};

// Composants stylisés
const FlowContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  height: 600,
  background: 'linear-gradient(to bottom, #f8f9fa, #eaeff1)',
  borderRadius: 16,
  boxShadow: 'inset 0 0 20px rgba(0,0,0,0.05)',
  overflow: 'hidden'
}));

const TaskSettingsCard = styled(Card)(({ theme }) => ({
  position: 'absolute',
  zIndex: 10,
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 500,
  maxWidth: '90%',
  maxHeight: '80vh',
  overflow: 'auto',
  borderRadius: 16,
  boxShadow: '0 24px 48px rgba(0, 0, 0, 0.2)'
}));

// Composant principal
const WorkflowEditor = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const isEditMode = Boolean(id);

  // États généraux du workflow
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(isEditMode);
  const [saving, setSaving] = useState(false);
  const [workflowData, setWorkflowData] = useState({
    name: '',
    description: '',
    workflow_type: 'MATRIX_ADDITION',
    priority: 2,
    min_volunteers: 1,
    max_volunteers: 10,
    volunteer_preferences: [],
    tags: [],
    metadata: {
      input: {}
    }
  });
  
  // États pour les nœuds et les connexions ReactFlow
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  
  // États pour les ressources et paramètres avancés
  const [resources, setResources] = useState({
    cpu: 2,
    memory: 4,
    storage: 10
  });
  const [advancedSettings, setAdvancedSettings] = useState({
    allowTaskReassignment: true,
    optimizeResourceAllocation: true,
    maxExecutionTime: 3600, // 1 heure en secondes
    retryCount: 3
  });
  
  // États pour la gestion des matrices (pour les workflows de type matrice)
  const [matrixSettings, setMatrixSettings] = useState({
    matrixADimensions: [100, 100],
    matrixBDimensions: [100, 100],
    algorithm: 'standard',
    precision: 'double',
    blockSize: 100
  });
  
  // États pour les dialogues
  const [taskSettingsOpen, setTaskSettingsOpen] = useState(false);
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  
  // Chargement initial des données si en mode édition
  useEffect(() => {
    if (isEditMode) {
      const loadWorkflow = async () => {
        try {
          setLoading(true);
          const data = await fetchWorkflowDetails(id);
          
          // Mise à jour des données générales du workflow
          setWorkflowData({
            name: data.name || '',
            description: data.description || '',
            workflow_type: data.workflow_type || 'MATRIX_ADDITION',
            priority: data.priority || 2,
            min_volunteers: data.min_volunteers || 1,
            max_volunteers: data.max_volunteers || 10,
            volunteer_preferences: data.volunteer_preferences || [],
            tags: data.tags || [],
            metadata: data.metadata || { input: {} }
          });
          
          // Extraction des paramètres spécifiques aux matrices si applicable
          if (data.workflow_type === 'MATRIX_ADDITION' || data.workflow_type === 'MATRIX_MULTIPLICATION') {
            if (data.metadata && data.metadata.input) {
              const input = data.metadata.input;
              setMatrixSettings({
                matrixADimensions: input.matrix_a?.dimensions || [100, 100],
                matrixBDimensions: input.matrix_b?.dimensions || [100, 100],
                algorithm: input.algorithm || 'standard',
                precision: input.precision || 'double',
                blockSize: input.block_size || 100
              });
            }
          }
          
          // Extraction des ressources estimées
          if (data.estimated_resources) {
            setResources({
              cpu: data.estimated_resources.cpu || 2,
              memory: data.estimated_resources.memory || 4,
              storage: data.estimated_resources.storage || 10
            });
          }
          
          // Paramètres avancés
          setAdvancedSettings({
            allowTaskReassignment: data.retry_count > 0,
            optimizeResourceAllocation: true,
            maxExecutionTime: data.max_execution_time || 3600,
            retryCount: data.retry_count || 3
          });
          
          // Création des nœuds pour ReactFlow basés sur les tâches
          if (data.tasks && Array.isArray(data.tasks)) {
            const workflowNodes = data.tasks.map((task, index) => ({
              id: task.id,
              type: 'custom',
              position: { x: 100 + (index % 3) * 250, y: 100 + Math.floor(index / 3) * 150 },
              data: {
                label: task.name || `Tâche ${index + 1}`,
                type: getTaskType(task),
                color: getTaskColor(task),
                parameters: task.parameters || {},
                onSettings: handleOpenTaskSettings,
                onViewCode: handleViewTaskCode,
                onPreview: handlePreviewTask,
                onDelete: handleDeleteTask
              }
            }));
            
            setNodes(workflowNodes);
            
            // Création des connexions basées sur les dépendances
            const workflowEdges = [];
            data.tasks.forEach(task => {
              if (task.dependencies && Array.isArray(task.dependencies)) {
                task.dependencies.forEach(depId => {
                  workflowEdges.push({
                    id: `edge-${depId}-${task.id}`,
                    source: depId,
                    target: task.id,
                    animated: true,
                    style: { stroke: '#0A2463', strokeWidth: 2 }
                  });
                });
              }
            });
            
            setEdges(workflowEdges);
          }
          
          setLoading(false);
        } catch (error) {
          console.error('Erreur lors du chargement du workflow:', error);
          showSnackbar('Erreur lors du chargement du workflow', 'error');
          setLoading(false);
        }
      };
      
      loadWorkflow();
    }
  }, [id, isEditMode]);
  
  // Gestionnaire pour les connexions
  const onConnect = useCallback((connection) => {
    setEdges((eds) => addEdge({
      ...connection,
      animated: true,
      style: { stroke: '#0A2463', strokeWidth: 2 }
    }, eds));
  }, [setEdges]);

  // Gestionnaire pour l'ajout d'une nouvelle tâche
  const addNewTask = (type) => {
    const newNodeId = `task-${Date.now()}`;
    const nodeColor = {
      'processing': '#4CAF50',
      'data': '#2196F3',
      'analysis': '#9C27B0',
      'output': '#D90429'
    }[type] || '#0A2463';
    
    const newNode = {
      id: newNodeId,
      type: 'custom',
      position: {
        x: 200 + Math.random() * 200,
        y: 100 + Math.random() * 200
      },
      data: {
        label: `Tâche ${nodes.length + 1}`,
        type: type.charAt(0).toUpperCase() + type.slice(1),
        color: nodeColor,
        parameters: {},
        onSettings: handleOpenTaskSettings,
        onViewCode: handleViewTaskCode,
        onPreview: handlePreviewTask,
        onDelete: handleDeleteTask
      }
    };
    
    setNodes([...nodes, newNode]);
  };
  
  // Gestionnaires pour les dialogues de tâches
  const handleOpenTaskSettings = (nodeId) => {
    setCurrentTaskId(nodeId);
    setTaskSettingsOpen(true);
  };
  
  const handleCloseTaskSettings = () => {
    setTaskSettingsOpen(false);
    setCurrentTaskId(null);
  };
  
  const handleViewTaskCode = (nodeId) => {
    // Logique pour afficher le code de la tâche
    console.log(`Afficher le code pour la tâche ${nodeId}`);
  };
  
  const handlePreviewTask = (nodeId) => {
    // Logique pour prévisualiser la tâche
    console.log(`Prévisualiser la tâche ${nodeId}`);
  };
  
  const handleDeleteTask = (nodeId) => {
    // Supprime le nœud
    setNodes(nodes.filter(node => node.id !== nodeId));
    
    // Supprime toutes les connexions associées à ce nœud
    setEdges(edges.filter(edge => edge.source !== nodeId && edge.target !== nodeId));
  };
  
  // Gestionnaire pour sauvegarder le workflow
  const handleSaveWorkflow = async () => {
    if (!workflowData.name) {
      showSnackbar('Veuillez donner un nom au workflow', 'error');
      return;
    }
    
    try {
      setSaving(true);
      
      // Préparation des données pour l'API
      const workflowToSave = {
        name: workflowData.name,
        description: workflowData.description,
        workflow_type: workflowData.workflow_type,
        tags: workflowData.tags,
        min_volunteers: workflowData.min_volunteers,
        max_volunteers: workflowData.max_volunteers,
        volunteer_preferences: workflowData.volunteer_preferences,
        priority: workflowData.priority,
        estimated_resources: {
          cpu: resources.cpu,
          memory: resources.memory,
          storage: resources.storage
        },
        max_execution_time: advancedSettings.maxExecutionTime,
        retry_count: advancedSettings.retryCount
      };
      
      // Paramètres spécifiques au type de workflow
      if (workflowData.workflow_type === 'MATRIX_ADDITION' || workflowData.workflow_type === 'MATRIX_MULTIPLICATION') {
        workflowToSave.input_data = {
          algorithm: matrixSettings.algorithm,
          precision: matrixSettings.precision,
          block_size: matrixSettings.blockSize,
          matrix_a: {
            format: "numpy",
            dimensions: matrixSettings.matrixADimensions,
            storage_type: "embedded"
          },
          matrix_b: {
            format: "numpy",
            dimensions: matrixSettings.matrixBDimensions,
            storage_type: "embedded"
          }
        };
      }
      
      let savedWorkflow;
      
      if (isEditMode) {
        // Logique pour mettre à jour un workflow existant
        // À implémenter selon votre API
        savedWorkflow = { id };
        showSnackbar('Workflow mis à jour avec succès', 'success');
      } else {
        // Créer un nouveau workflow
        savedWorkflow = await createMatrixWorkflow(workflowToSave);
        showSnackbar('Workflow créé avec succès', 'success');
      }
      
      // Redirection vers la page de détails du workflow
      setTimeout(() => {
        navigate(`/status/${savedWorkflow.id}`);
      }, 1500);
    } catch (error) {
      console.error('Erreur lors de la sauvegarde du workflow:', error);
      showSnackbar('Erreur lors de la sauvegarde du workflow', 'error');
    } finally {
      setSaving(false);
    }
  };
  
  // Gestionnaire pour exécuter le workflow
  const handleExecuteWorkflow = async () => {
    try {
      setSaving(true);
      
      // D'abord, sauvegarder le workflow
      await handleSaveWorkflow();
      
      // Puis, soumettre le workflow pour exécution
      if (id) {
        await submitWorkflow(id);
        showSnackbar('Workflow soumis pour exécution', 'success');
        
        // Redirection vers la page de statut du workflow
        setTimeout(() => {
          navigate(`/status/${id}`);
        }, 1500);
      }
    } catch (error) {
      console.error('Erreur lors de l\'exécution du workflow:', error);
      showSnackbar('Erreur lors de l\'exécution du workflow', 'error');
    } finally {
      setSaving(false);
    }
  };
  
  // Utilitaire pour afficher des notifications
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
  
  // Utilitaires pour les tâches
  const getTaskType = (task) => {
    // Détermination du type de tâche basée sur son rôle ou ses paramètres
    if (task.type) return task.type;
    if (task.is_output) return 'Output';
    if (task.is_input) return 'Data';
    if (task.is_analysis) return 'Analysis';
    return 'Processing';
  };
  
  const getTaskColor = (task) => {
    const taskType = getTaskType(task);
    const colorMap = {
      'Processing': '#4CAF50',
      'Data': '#2196F3',
      'Analysis': '#9C27B0',
      'Output': '#D90429'
    };
    return colorMap[taskType] || '#0A2463';
  };
  
  // Gestion des changements d'état
  const handleWorkflowDataChange = (field) => (event) => {
    setWorkflowData({
      ...workflowData,
      [field]: event.target.value
    });
  };
  
  const handleMatrixSettingsChange = (field) => (event) => {
    setMatrixSettings({
      ...matrixSettings,
      [field]: event.target.value
    });
  };
  
  const handleMatrixDimensionChange = (matrix, index) => (event) => {
    const value = parseInt(event.target.value) || 10;
    const dimensionField = `${matrix}Dimensions`;
    const newDimensions = [...matrixSettings[dimensionField]];
    newDimensions[index] = value;
    
    // Si c'est une multiplication de matrices, ajuster automatiquement les dimensions
    if (workflowData.workflow_type === 'MATRIX_MULTIPLICATION' && matrix === 'matrixA' && index === 1) {
      // La colonne de A détermine la ligne de B
      const newBDimensions = [...matrixSettings.matrixBDimensions];
      newBDimensions[0] = value;
      
      setMatrixSettings({
        ...matrixSettings,
        [dimensionField]: newDimensions,
        matrixBDimensions: newBDimensions
      });
    } else if (workflowData.workflow_type === 'MATRIX_ADDITION') {
      // Pour l'addition, les deux matrices doivent avoir les mêmes dimensions
      if (matrix === 'matrixA') {
        const newBDimensions = [...matrixSettings.matrixBDimensions];
        newBDimensions[index] = value;
        
        setMatrixSettings({
          ...matrixSettings,
          [dimensionField]: newDimensions,
          matrixBDimensions: newBDimensions
        });
      } else {
        const newADimensions = [...matrixSettings.matrixADimensions];
        newADimensions[index] = value;
        
        setMatrixSettings({
          ...matrixSettings,
          [dimensionField]: newDimensions,
          matrixADimensions: newADimensions
        });
      }
    } else {
      setMatrixSettings({
        ...matrixSettings,
        [dimensionField]: newDimensions
      });
    }
  };
  
  const handleResourceChange = (field) => (event) => {
    setResources({
      ...resources,
      [field]: event.target.value
    });
  };
  
  const handleAdvancedSettingChange = (field) => (event) => {
    setAdvancedSettings({
      ...advancedSettings,
      [field]: field === 'allowTaskReassignment' || field === 'optimizeResourceAllocation' ? 
        event.target.checked : event.target.value
    });
  };
  
  const handleTagChange = (event, newValue) => {
    setWorkflowData({
      ...workflowData,
      tags: newValue
    });
  };
  
  const handleVolunteerPreferenceChange = (event, newValue) => {
    setWorkflowData({
      ...workflowData,
      volunteer_preferences: newValue
    });
  };

  // Configuration des types de nœuds personnalisés
  const nodeTypes = {
    custom: CustomTaskNode
  };
  
  // Options pour les champs select
  const volunteerTypes = [
    'ANY', 'CPU_INTENSIVE', 'GPU_REQUIRED', 'NETWORK_INTENSIVE', 'LOW_RESOURCE', 'HIGH_MEMORY'
  ];
  
  const algorithms = [
    { value: 'standard', label: 'Standard' },
    { value: 'strassen', label: 'Strassen' },
    { value: 'block', label: 'Par blocs' }
  ];
  
  const precisionTypes = [
    { value: 'single', label: 'Simple (float)' },
    { value: 'double', label: 'Double (double)' },
    { value: 'integer', label: 'Entier (int)' }
  ];

  // Rendu pendant le chargement
  if (loading) {
    return (
      <Backdrop open={true} sx={{ color: '#fff', zIndex: 1000 }}>
        <CircularProgress color="inherit" />
      </Backdrop>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Overlay de sauvegarde */}
      <Backdrop open={saving} sx={{ color: '#fff', zIndex: 1300 }}>
        <Box sx={{ textAlign: 'center' }}>
          <CircularProgress color="inherit" sx={{ mb: 2 }} />
          <Typography variant="h6" color="white">
            {isEditMode ? 'Mise à jour du workflow...' : 'Création du workflow...'}
          </Typography>
        </Box>
      </Backdrop>
      
      {/* En-tête avec actions principales */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton onClick={() => navigate(-1)} sx={{ mr: 2 }}>
            <ArrowBack />
          </IconButton>
          <Typography variant="h4" fontWeight="bold">
            {isEditMode ? "Modifier le workflow" : "Créer un nouveau workflow"}
          </Typography>
        </Box>
        
        <Box>
          <Button
            variant="contained"
            color="primary"
            startIcon={<Save />}
            sx={{ mr: 1 }}
            onClick={handleSaveWorkflow}
            disabled={saving}
          >
            Enregistrer
          </Button>
          <Button
            variant="contained"
            color="secondary"
            startIcon={<PlayArrow />}
            onClick={handleExecuteWorkflow}
            disabled={saving}
          >
            Exécuter
          </Button>
        </Box>
      </Box>

      {/* Onglets de l'éditeur */}
      <Tabs 
        value={activeTab} 
        onChange={(e, v) => setActiveTab(v)}
        sx={{ mb: 3 }}
      >
        <Tab label="Informations" />
        <Tab label="Configuration des matrices" 
          disabled={!['MATRIX_ADDITION', 'MATRIX_MULTIPLICATION'].includes(workflowData.workflow_type)} 
        />
        <Tab label="Tâches et dépendances" />
        <Tab label="Paramètres avancés" />
      </Tabs>

      {/* Contenu de l'onglet Informations */}
      <Box sx={{ display: activeTab === 0 ? 'block' : 'none' }}>
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Nom du workflow"
                  variant="outlined"
                  fullWidth
                  value={workflowData.name}
                  onChange={handleWorkflowDataChange('name')}
                  required
                  error={!workflowData.name}
                  helperText={!workflowData.name ? "Le nom est obligatoire" : ""}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Type de workflow</InputLabel>
                  <Select
                    value={workflowData.workflow_type}
                    label="Type de workflow"
                    onChange={handleWorkflowDataChange('workflow_type')}
                  >
                    <MenuItem value="MATRIX_ADDITION">Addition de matrices</MenuItem>
                    <MenuItem value="MATRIX_MULTIPLICATION">Multiplication de matrices</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Description"
                  variant="outlined"
                  fullWidth
                  multiline
                  rows={4}
                  value={workflowData.description}
                  onChange={handleWorkflowDataChange('description')}
                  placeholder="Décrivez l'objectif et les caractéristiques de ce workflow..."
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>Priorité</Typography>
                <Slider
                  value={workflowData.priority}
                  onChange={(e, v) => setWorkflowData({...workflowData, priority: v})}
                  step={1}
                  marks
                  min={1}
                  max={5}
                  valueLabelDisplay="auto"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Autocomplete
                  multiple
                  options={[]}
                  value={workflowData.tags}
                  onChange={handleTagChange}
                  freeSolo
                  renderTags={(value, getTagProps) =>
                    value.map((option, index) => (
                      <Chip 
                        label={option} 
                        {...getTagProps({ index })} 
                        color="primary" 
                        variant="outlined"
                      />
                    ))
                  }
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      label="Tags"
                      placeholder="Ajouter un tag"
                      helperText="Appuyez sur Entrée pour ajouter un tag"
                    />
                  )}
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Configuration des matrices</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              {workflowData.workflow_type === 'MATRIX_ADDITION' ? 
                "Définissez les dimensions et les paramètres pour l'addition de matrices" : 
                "Définissez les dimensions et les paramètres pour la multiplication de matrices"}
            </Typography>
            
            <Grid container spacing={3}>
              {/* Matrice A */}
              <Grid item xs={12}>
                <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
                  Matrice A
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Nombre de lignes"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={matrixSettings.matrixADimensions[0]}
                  onChange={handleMatrixDimensionChange('matrixA', 0)}
                  InputProps={{ inputProps: { min: 2, max: 1000 } }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Nombre de colonnes"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={matrixSettings.matrixADimensions[1]}
                  onChange={handleMatrixDimensionChange('matrixA', 1)}
                  InputProps={{ inputProps: { min: 2, max: 1000 } }}
                />
              </Grid>
              
              {/* Matrice B */}
              <Grid item xs={12}>
                <Typography variant="subtitle1" fontWeight="medium" gutterBottom sx={{ mt: 2 }}>
                  Matrice B
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Nombre de lignes"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={matrixSettings.matrixBDimensions[0]}
                  onChange={handleMatrixDimensionChange('matrixB', 0)}
                  InputProps={{ 
                    inputProps: { min: 2, max: 1000 },
                    readOnly: workflowData.workflow_type === 'MATRIX_MULTIPLICATION'
                  }}
                  disabled={workflowData.workflow_type === 'MATRIX_MULTIPLICATION'}
                  helperText={workflowData.workflow_type === 'MATRIX_MULTIPLICATION' ? 
                    "Doit être égal au nombre de colonnes de la matrice A" : ""}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Nombre de colonnes"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={matrixSettings.matrixBDimensions[1]}
                  onChange={handleMatrixDimensionChange('matrixB', 1)}
                  InputProps={{ 
                    inputProps: { min: 2, max: 1000 },
                    readOnly: workflowData.workflow_type === 'MATRIX_ADDITION'
                  }}
                  disabled={workflowData.workflow_type === 'MATRIX_ADDITION'}
                  helperText={workflowData.workflow_type === 'MATRIX_ADDITION' ? 
                    "Doit être égal au nombre de colonnes de la matrice A" : ""}
                />
              </Grid>
              
              {/* Paramètres de calcul */}
              <Grid item xs={12}>
                <Divider sx={{ my: 3 }} />
                <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
                  Paramètres de calcul
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel>Algorithme</InputLabel>
                  <Select
                    value={matrixSettings.algorithm}
                    label="Algorithme"
                    onChange={handleMatrixSettingsChange('algorithm')}
                  >
                    {algorithms.map(algo => (
                      <MenuItem key={algo.value} value={algo.value}>{algo.label}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel>Précision</InputLabel>
                  <Select
                    value={matrixSettings.precision}
                    label="Précision"
                    onChange={handleMatrixSettingsChange('precision')}
                  >
                    {precisionTypes.map(precision => (
                      <MenuItem key={precision.value} value={precision.value}>{precision.label}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Taille des blocs"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={matrixSettings.blockSize}
                  onChange={handleMatrixSettingsChange('blockSize')}
                  InputProps={{ inputProps: { min: 10, max: 500 } }}
                  helperText="Pour l'algorithme par blocs"
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>

      {/* Contenu de l'onglet Tâches et dépendances */}
      <Box sx={{ display: activeTab === 2 ? 'block' : 'none' }}>
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Conception du workflow</Typography>
              <Box>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Add />}
                  onClick={() => addNewTask('processing')}
                  sx={{ mr: 1 }}
                >
                  Traitement
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Add />}
                  onClick={() => addNewTask('data')}
                  sx={{ mr: 1 }}
                >
                  Données
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Add />}
                  onClick={() => addNewTask('analysis')}
                  sx={{ mr: 1 }}
                >
                  Analyse
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Add />}
                  onClick={() => addNewTask('output')}
                >
                  Sortie
                </Button>
              </Box>
            </Box>

            <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
              Glissez pour positionner les tâches et connectez-les pour définir les dépendances.
              Cliquez et faites glisser depuis un nœud vers un autre pour créer une connexion.
            </Typography>

            <FlowContainer>
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                fitView
              >
                <Background />
                <Controls />
                <MiniMap />
              </ReactFlow>
            </FlowContainer>
          </CardContent>
        </Card>
      </Box>

      {/* Contenu de l'onglet Paramètres avancés */}
      <Box sx={{ display: activeTab === 3 ? 'block' : 'none' }}>
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Paramètres des ressources</Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <TextField
                  label="CPU requis (cœurs)"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={resources.cpu}
                  onChange={handleResourceChange('cpu')}
                  InputProps={{ inputProps: { min: 1 } }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Mémoire (GB)"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={resources.memory}
                  onChange={handleResourceChange('memory')}
                  InputProps={{ inputProps: { min: 1 } }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Stockage (GB)"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={resources.storage}
                  onChange={handleResourceChange('storage')}
                  InputProps={{ inputProps: { min: 1 } }}
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Options d'exécution</Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Temps d'exécution maximum (secondes)"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={advancedSettings.maxExecutionTime}
                  onChange={handleAdvancedSettingChange('maxExecutionTime')}
                  InputProps={{ inputProps: { min: 60 } }}
                  helperText="Durée maximale avant arrêt forcé (en secondes)"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Nombre de tentatives en cas d'échec"
                  type="number"
                  variant="outlined"
                  fullWidth
                  value={advancedSettings.retryCount}
                  onChange={handleAdvancedSettingChange('retryCount')}
                  InputProps={{ inputProps: { min: 0, max: 10 } }}
                  helperText="Nombre de réessais avant échec définitif"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch 
                      checked={advancedSettings.allowTaskReassignment}
                      onChange={handleAdvancedSettingChange('allowTaskReassignment')}
                    />
                  }
                  label="Autoriser la redistribution des tâches en cas de déconnexion"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch 
                      checked={advancedSettings.optimizeResourceAllocation}
                      onChange={handleAdvancedSettingChange('optimizeResourceAllocation')}
                    />
                  }
                  label="Optimiser automatiquement les allocations de ressources"
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>

      {/* Dialogue de paramétrage des tâches */}
      {taskSettingsOpen && currentTaskId && (
        <TaskSettingsCard>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6">Paramètres de la tâche</Typography>
              <IconButton onClick={handleCloseTaskSettings}>
                <Delete />
              </IconButton>
            </Box>
            
            <TextField
              label="Nom de la tâche"
              variant="outlined"
              fullWidth
              defaultValue={nodes.find(node => node.id === currentTaskId)?.data.label || ""}
              onChange={(e) => {
                setNodes(nodes.map(node => 
                  node.id === currentTaskId 
                    ? { ...node, data: { ...node.data, label: e.target.value } }
                    : node
                ));
              }}
              sx={{ mb: 3 }}
            />
            
            <Typography variant="subtitle1" gutterBottom>Type de tâche</Typography>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <Select
                value={nodes.find(node => node.id === currentTaskId)?.data.type || "Processing"}
                onChange={(e) => {
                  const taskType = e.target.value;
                  const nodeColor = {
                    'Processing': '#4CAF50',
                    'Data': '#2196F3',
                    'Analysis': '#9C27B0',
                    'Output': '#D90429'
                  }[taskType] || '#0A2463';
                  
                  setNodes(nodes.map(node => 
                    node.id === currentTaskId 
                      ? { 
                          ...node, 
                          data: { 
                            ...node.data, 
                            type: taskType,
                            color: nodeColor
                          } 
                        }
                      : node
                  ));
                }}
              >
                <MenuItem value="Processing">Traitement</MenuItem>
                <MenuItem value="Data">Données</MenuItem>
                <MenuItem value="Analysis">Analyse</MenuItem>
                <MenuItem value="Output">Sortie</MenuItem>
              </Select>
            </FormControl>
            
            <Button
              variant="contained"
              color="primary"
              fullWidth
              onClick={handleCloseTaskSettings}
            >
              Appliquer
            </Button>
          </CardContent>
        </TaskSettingsCard>
      )}
      
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

export default WorkflowEditor;