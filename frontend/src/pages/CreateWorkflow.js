import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Typography, Container, Button, TextField, Paper,
  Card, CardContent, IconButton, List, ListItem, ListItemText,
  Divider, Chip, Stack, Alert, Snackbar, MenuItem, FormControl,
  Select, InputLabel, FormControlLabel, Checkbox, Slider
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  CloudUpload, Add, Delete, ArrowForward, 
  Description, Tag, Category, Calculate
} from '@mui/icons-material';
import { 
  createMatrixWorkflow, 
  submitWorkflow,
  generateRandomMatrix
} from '../services/api';

// Styled components
const StyledContainer = styled(Container)(({ theme }) => ({
  padding: theme.spacing(4),
}));

const FormCard = styled(Card)(({ theme }) => ({
  borderRadius: 16,
  background: 'linear-gradient(135deg, #0A1929 0%, #0E1D35 100%)',
  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.25)',
  overflow: 'visible',
  border: '1px solid rgba(76, 175, 80, 0.2)',
  transition: 'transform 0.3s ease',
  '&:hover': {
    boxShadow: '0 15px 35px rgba(0, 0, 0, 0.3)',
    transform: 'translateY(-5px)',
  }
}));

const CardTitle = styled(Typography)(({ theme }) => ({
  fontSize: '1.5rem',
  fontWeight: 600,
  marginBottom: theme.spacing(3),
  color: '#fff',
  textAlign: 'center',
  position: 'relative',
  '&:after': {
    content: '""',
    position: 'absolute',
    bottom: -10,
    left: '50%',
    transform: 'translateX(-50%)',
    height: 3,
    width: 80,
    background: 'linear-gradient(90deg, #4CAF50, transparent)',
    borderRadius: 3,
  }
}));

const StyledButton = styled(Button)(({ theme }) => ({
  borderRadius: 50,
  padding: '12px 30px',
  fontSize: '1rem',
  fontWeight: 600,
  textTransform: 'none',
  transition: 'all 0.3s ease',
  background: 'linear-gradient(45deg, #357a38 30%, #4CAF50 90%)',
  boxShadow: '0 5px 15px rgba(76, 175, 80, 0.4)',
  '&:hover': {
    boxShadow: '0 8px 20px rgba(76, 175, 80, 0.6)',
    transform: 'translateY(-2px)',
  }
}));

const UploadButton = styled(Button)(({ theme }) => ({
  width: '100%',
  height: 120,
  borderRadius: 12,
  border: '2px dashed rgba(76, 175, 80, 0.5)',
  background: 'rgba(76, 175, 80, 0.05)',
  transition: 'all 0.3s ease',
  '&:hover': {
    background: 'rgba(76, 175, 80, 0.1)',
    borderColor: 'rgba(76, 175, 80, 0.8)',
  }
}));

const StyledChip = styled(Chip)(({ theme }) => ({
  background: 'rgba(76, 175, 80, 0.15)',
  border: '1px solid rgba(76, 175, 80, 0.3)',
  color: '#fff',
  margin: theme.spacing(0.5),
  '&:hover': {
    background: 'rgba(76, 175, 80, 0.25)',
  }
}));

const StyledTextField = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    '&:hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.08)',
    },
    '&.Mui-focused': {
      boxShadow: '0 0 15px rgba(76, 175, 80, 0.2)',
    }
  },
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: 'rgba(255, 255, 255, 0.1)',
  }
}));

const StyledSelect = styled(Select)(({ theme }) => ({
  borderRadius: 12,
  backgroundColor: 'rgba(255, 255, 255, 0.05)',
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
  },
  '&.Mui-focused': {
    boxShadow: '0 0 15px rgba(76, 175, 80, 0.2)',
  },
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: 'rgba(255, 255, 255, 0.1)',
  }
}));

const FileItem = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1.5),
  borderRadius: 10,
  display: 'flex',
  alignItems: 'center',
  marginBottom: theme.spacing(1.5),
  background: 'rgba(255, 255, 255, 0.05)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  '&:hover': {
    background: 'rgba(255, 255, 255, 0.08)',
    boxShadow: '0 4px 10px rgba(0, 0, 0, 0.15)',
  }
}));

const PageTitle = styled(Typography)(({ theme }) => ({
  fontSize: '2rem',
  fontWeight: 700,
  marginBottom: theme.spacing(4),
  textAlign: 'center',
  background: 'linear-gradient(45deg, #FFFFFF 30%, #4CAF50 90%)',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  filter: 'drop-shadow(0 2px 5px rgba(0, 0, 0, 0.5))'
}));

// Définir les types de workflows avec leurs correspondances backend
const workflowTypes = [
  { display: "Addition de matrices", value: "MATRIX_ADDITION" },
  { display: "Multiplication de matrices", value: "MATRIX_MULTIPLICATION" }
];

// Algorithmes disponibles
const matrixAlgorithms = [
  { display: "Standard", value: "standard" },
  { display: "Strassen", value: "strassen" },
  { display: "Par blocs", value: "block" }
];

// Formats de matrices
const matrixFormats = [
  { display: "NumPy", value: "numpy" },
  { display: "CSV", value: "csv" },
  { display: "JSON", value: "json" }
];

// Types de précision
const precisionTypes = [
  { display: "Simple (float)", value: "single" },
  { display: "Double (double)", value: "double" },
  { display: "Entier (int)", value: "integer" }
];

// Préférences de volontaires
const volunteerTypes = [
  { display: "Tous types", value: "ANY" },
  { display: "Calcul intensif CPU", value: "CPU_INTENSIVE" },
  { display: "GPU nécessaire", value: "GPU_REQUIRED" },
  { display: "Transfert intensif réseau", value: "NETWORK_INTENSIVE" },
  { display: "Ressources limitées", value: "LOW_RESOURCE" },
  { display: "Mémoire importante", value: "HIGH_MEMORY" }
];

// Main component
const CreateMatrixWorkflow = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [currentTag, setCurrentTag] = useState('');
  const [currentPreference, setCurrentPreference] = useState('');
  
  // État initial pour le workflow matriciel
  const [workflowData, setWorkflowData] = useState({
    name: '',
    description: '',
    type: workflowTypes[0].value, // Valeur par défaut: MATRIX_ADDITION
    displayType: workflowTypes[0].display,
    tags: [],
    
    // Paramètres pour la matrice A
    matrixAFormat: 'numpy', 
    matrixADimensions: [100, 100],
    matrixAData: null,
    matrixAFile: null,
    
    // Paramètres pour la matrice B
    matrixBFormat: 'numpy',
    matrixBDimensions: [100, 100],
    matrixBData: null,
    matrixBFile: null,
    
    // Paramètres de calcul
    algorithm: 'standard',
    precision: 'double',
    blockSize: 100,
    
    // Paramètres de volontaires
    minVolunteers: 1,
    maxVolunteers: 10,
    volunteerPreferences: [],
    
    // Mode de génération
    generationMode: 'random', // 'random', 'upload' ou 'manual'
  });

  // Gestion du téléchargement des fichiers
  const [selectedFileA, setSelectedFileA] = useState(null);
  const [selectedFileB, setSelectedFileB] = useState(null);

  // Handle change for basic fields
  const handleChange = (e) => {
    const { name, value } = e.target;
    
    // Cas spécial pour le changement de type
    if (name === 'type') {
      const selectedType = workflowTypes.find(t => t.value === value);
      setWorkflowData({
        ...workflowData,
        type: value,
        displayType: selectedType.display
      });
    } else {
      setWorkflowData({
        ...workflowData,
        [name]: value
      });
    }
  };

  // Gestion des dimensions de matrice
  const handleDimensionChange = (matrix, index, value) => {
    // Valider que c'est un nombre positif
    const numValue = parseInt(value, 10) || 10;
    const validValue = Math.max(2, Math.min(1000, numValue));
    
    const dimensionKey = `${matrix}Dimensions`;
    const newDimensions = [...workflowData[dimensionKey]];
    newDimensions[index] = validValue;
    
    // Ajuster automatiquement les dimensions selon le type d'opération
    if (workflowData.type === 'MATRIX_MULTIPLICATION' && matrix === 'matrixA' && index === 1) {
      // Pour multiplication: si cols de A change, rows de B doit correspondre
      const newBDimensions = [...workflowData.matrixBDimensions];
      newBDimensions[0] = validValue;
      
      setWorkflowData({
        ...workflowData,
        [dimensionKey]: newDimensions,
        matrixBDimensions: newBDimensions
      });
    } else {
      setWorkflowData({
        ...workflowData,
        [dimensionKey]: newDimensions
      });
    }
    
    // Si en mode aléatoire, régénérer les matrices avec nouvelles dimensions
    if (workflowData.generationMode === 'random') {
      regenerateRandomMatrices();
    }
  };

  // Fonction pour générer des matrices aléatoires
  const regenerateRandomMatrices = () => {
    const matrixAData = generateRandomMatrix(workflowData.matrixADimensions);
    let matrixBData;
    
    if (workflowData.type === 'MATRIX_ADDITION') {
      // Pour addition, les dimensions doivent être identiques
      matrixBData = generateRandomMatrix(workflowData.matrixADimensions);
      
      setWorkflowData({
        ...workflowData,
        matrixAData,
        matrixBData,
        matrixBDimensions: [...workflowData.matrixADimensions]
      });
    } else {
      // Pour multiplication, les dimensions B doivent être [colsA, ?]
      const colsA = workflowData.matrixADimensions[1];
      const newBDimensions = [colsA, workflowData.matrixBDimensions[1]];
      matrixBData = generateRandomMatrix(newBDimensions);
      
      setWorkflowData({
        ...workflowData,
        matrixAData,
        matrixBData,
        matrixBDimensions: newBDimensions
      });
    }
  };

  // Changer le mode de génération
  const handleGenerationModeChange = (mode) => {
    setWorkflowData({
      ...workflowData,
      generationMode: mode
    });
    
    if (mode === 'random') {
      regenerateRandomMatrices();
    }
  };

  // Gestion des tags
  const handleAddTag = () => {
    if (currentTag && !workflowData.tags.includes(currentTag)) {
      setWorkflowData({
        ...workflowData,
        tags: [...workflowData.tags, currentTag]
      });
      setCurrentTag('');
    }
  };

  const handleRemoveTag = (tag) => {
    setWorkflowData({
      ...workflowData,
      tags: workflowData.tags.filter(t => t !== tag)
    });
  };

  // Gestion des préférences de volontaires
  const handleAddPreference = () => {
    if (currentPreference && !workflowData.volunteerPreferences.includes(currentPreference)) {
      setWorkflowData({
        ...workflowData,
        volunteerPreferences: [...workflowData.volunteerPreferences, currentPreference]
      });
      setCurrentPreference('');
    }
  };

  const handleRemovePreference = (pref) => {
    setWorkflowData({
      ...workflowData,
      volunteerPreferences: workflowData.volunteerPreferences.filter(p => p !== pref)
    });
  };

  // Gestion des fichiers
  const handleFileChange = (matrix, e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const fileKey = `${matrix}File`;
    const fileSetterKey = matrix === 'matrixA' ? setSelectedFileA : setSelectedFileB;
    
    // Vérifier le type de fichier (uniquement CSV, JSON, ou NPY)
    const allowedTypes = ['.csv', '.json', '.npy', '.txt'];
    const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    
    if (!allowedTypes.includes(fileExt)) {
      setError(`Format de fichier non supporté. Formats acceptés: ${allowedTypes.join(', ')}`);
      return;
    }
    
    // Mettre à jour le fichier sélectionné et changer le format si nécessaire
    fileSetterKey(file);
    
    let format = 'numpy'; // Par défaut
    if (fileExt === '.csv') format = 'csv';
    if (fileExt === '.json') format = 'json';
    
    const formatKey = `${matrix}Format`;
    
    setWorkflowData({
      ...workflowData,
      [fileKey]: file,
      [formatKey]: format,
      generationMode: 'upload'
    });
  };

  // Navigation entre les étapes
  const goToNextStep = () => setCurrentStep(currentStep + 1);
  const goToPreviousStep = () => setCurrentStep(currentStep - 1);

  // Effet pour générer des matrices aléatoires au chargement initial
  useEffect(() => {
    if (workflowData.generationMode === 'random' && !workflowData.matrixAData) {
      regenerateRandomMatrices();
    }
  }, []);

  // Effet pour synchroniser les dimensions en mode Addition
  useEffect(() => {
    if (workflowData.type === 'MATRIX_ADDITION') {
      // Si les dimensions ne sont pas égales, synchroniser B avec A
      const dimA = workflowData.matrixADimensions;
      const dimB = workflowData.matrixBDimensions;
      
      if (dimA[0] !== dimB[0] || dimA[1] !== dimB[1]) {
        setWorkflowData({
          ...workflowData,
          matrixBDimensions: [...dimA]
        });
        
        // Régénérer la matrice B si en mode aléatoire
        if (workflowData.generationMode === 'random') {
          const matrixBData = generateRandomMatrix(dimA);
          setWorkflowData(prev => ({
            ...prev,
            matrixBData
          }));
        }
      }
    } else if (workflowData.type === 'MATRIX_MULTIPLICATION') {
      // Pour multiplication, vérifier que colonnes de A = lignes de B
      const colsA = workflowData.matrixADimensions[1];
      const rowsB = workflowData.matrixBDimensions[0];
      
      if (colsA !== rowsB) {
        const newBDimensions = [colsA, workflowData.matrixBDimensions[1]];
        setWorkflowData(prev => ({
          ...prev,
          matrixBDimensions: newBDimensions
        }));
        
        // Régénérer la matrice B si en mode aléatoire
        if (workflowData.generationMode === 'random') {
          const matrixBData = generateRandomMatrix(newBDimensions);
          setWorkflowData(prev => ({
            ...prev,
            matrixBData
          }));
        }
      }
    }
  }, [workflowData.type, workflowData.matrixADimensions]);

  // Prépare les données pour l'envoi au backend
  const prepareDataForSubmission = () => {
    return {
      name: workflowData.name,
      description: workflowData.description,
      type: workflowData.type,
      tags: workflowData.tags,
      
      // Matrices
      matrixAFormat: workflowData.matrixAFormat,
      matrixADimensions: workflowData.matrixADimensions,
      matrixAData: workflowData.matrixAData,
      
      matrixBFormat: workflowData.matrixBFormat,
      matrixBDimensions: workflowData.matrixBDimensions,
      matrixBData: workflowData.matrixBData,
      
      // Paramètres
      algorithm: workflowData.algorithm,
      precision: workflowData.precision,
      blockSize: workflowData.blockSize,
      
      // Volontaires
      minVolunteers: workflowData.minVolunteers,
      maxVolunteers: workflowData.maxVolunteers,
      volunteerPreferences: workflowData.volunteerPreferences
    };
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Valider les données essentielles
      if (!workflowData.name) {
        throw new Error("Le nom du workflow est obligatoire");
      }
      
      // Vérifier les matrices selon le mode de génération
      if (workflowData.generationMode === 'upload') {
        if (!selectedFileA || !selectedFileB) {
          throw new Error("Veuillez télécharger les deux matrices");
        }
      } else if (workflowData.generationMode === 'random') {
        if (!workflowData.matrixAData || !workflowData.matrixBData) {
          // Régénérer si nécessaire
          regenerateRandomMatrices();
        }
      }
      
      // Vérifier les dimensions pour multiplication
      if (workflowData.type === 'MATRIX_MULTIPLICATION') {
        const colsA = workflowData.matrixADimensions[1];
        const rowsB = workflowData.matrixBDimensions[0];
        
        if (colsA !== rowsB) {
          throw new Error("Pour la multiplication, le nombre de colonnes de A doit être égal au nombre de lignes de B");
        }
      }
      
      // Préparer les données pour la soumission
      const data = prepareDataForSubmission();
      
      // Créer le workflow
      const response = await createMatrixWorkflow(data);
      
      // Soumettre le workflow pour traitement
      await submitWorkflow(response.id);
      
      // Naviguer vers la page de détails
      navigate(`/workflows/${response.id}`);
    } catch (error) {
      console.error('Error creating workflow:', error);
      setError(error.message || "Une erreur s'est produite lors de la création du workflow");
      setLoading(false);
    }
  };

  // Render step content
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <Box>
            <CardTitle>Informations de base</CardTitle>
            <Stack spacing={3}>
              <StyledTextField
                fullWidth
                label="Nom du workflow"
                name="name"
                value={workflowData.name}
                onChange={handleChange}
                required
                InputProps={{
                  startAdornment: <Description sx={{ mr: 1, color: 'rgba(255,255,255,0.5)' }} />
                }}
              />
              
              <FormControl fullWidth>
                <InputLabel id="type-label">Type d'opération matricielle</InputLabel>
                <StyledSelect
                  labelId="type-label"
                  name="type"
                  value={workflowData.type}
                  onChange={handleChange}
                  label="Type d'opération matricielle"
                  InputProps={{
                    startAdornment: <Calculate sx={{ mr: 1, color: 'rgba(255,255,255,0.5)' }} />
                  }}
                >
                  {workflowTypes.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.display}
                    </MenuItem>
                  ))}
                </StyledSelect>
              </FormControl>
              
              <StyledTextField
                fullWidth
                label="Description"
                name="description"
                value={workflowData.description}
                onChange={handleChange}
                multiline
                rows={3}
                placeholder="Décrivez l'objectif de cette opération matricielle..."
              />
              
              <Box>
                <Box sx={{ display: 'flex', mb: 2 }}>
                  <StyledTextField
                    fullWidth
                    label="Tags"
                    value={currentTag}
                    onChange={(e) => setCurrentTag(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddTag()}
                    placeholder="Ajouter des mots-clés"
                    InputProps={{
                      startAdornment: <Tag sx={{ mr: 1, color: 'rgba(255,255,255,0.5)' }} />
                    }}
                  />
                  <IconButton 
                    onClick={handleAddTag} 
                    sx={{ ml: 1, color: '#4CAF50' }}
                  >
                    <Add />
                  </IconButton>
                </Box>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
                  {workflowData.tags.map(tag => (
                    <StyledChip
                      key={tag}
                      label={tag}
                      onDelete={() => handleRemoveTag(tag)}
                    />
                  ))}
                </Box>
              </Box>
            </Stack>
          </Box>
        );
        
      case 2:
        return (
          <Box>
            <CardTitle>Configuration des matrices</CardTitle>
            
            <Box sx={{ mb: 4 }}>
              <Typography variant="subtitle1" color="primary" gutterBottom>
                Mode de génération des matrices
              </Typography>
              
              <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={workflowData.generationMode === 'random'}
                      onChange={() => handleGenerationModeChange('random')}
                      sx={{ color: 'rgba(76, 175, 80, 0.8)' }}
                    />
                  }
                  label="Générer des matrices aléatoires"
                />
                
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={workflowData.generationMode === 'upload'}
                      onChange={() => handleGenerationModeChange('upload')}
                      sx={{ color: 'rgba(76, 175, 80, 0.8)' }}
                    />
                  }
                  label="Télécharger des fichiers"
                />
              </Stack>
              
              {workflowData.generationMode === 'random' && (
                <Box>
                  <Typography variant="subtitle1" color="primary" gutterBottom>
                    Dimensions des matrices
                  </Typography>
                  
                  <Stack spacing={2}>
                    <Typography variant="body1">Matrice A</Typography>
                    <Stack direction="row" spacing={2}>
                      <StyledTextField
                        label="Lignes"
                        type="number"
                        value={workflowData.matrixADimensions[0]}
                        onChange={(e) => handleDimensionChange('matrixA', 0, e.target.value)}
                        InputProps={{ inputProps: { min: 2, max: 1000 } }}
                      />
                      <StyledTextField
                        label="Colonnes"
                        type="number"
                        value={workflowData.matrixADimensions[1]}
                        onChange={(e) => handleDimensionChange('matrixA', 1, e.target.value)}
                        InputProps={{ inputProps: { min: 2, max: 1000 } }}
                      />
                    </Stack>
                    
                    <Typography variant="body1">Matrice B</Typography>
                    <Stack direction="row" spacing={2}>
                      <StyledTextField
                        label="Lignes"
                        type="number"
                        value={workflowData.matrixBDimensions[0]}
                        onChange={(e) => handleDimensionChange('matrixB', 0, e.target.value)}
                        InputProps={{ 
                          inputProps: { min: 2, max: 1000 },
                          readOnly: workflowData.type === 'MATRIX_MULTIPLICATION' 
                        }}
                        disabled={workflowData.type === 'MATRIX_MULTIPLICATION'}
                        helperText={workflowData.type === 'MATRIX_MULTIPLICATION' ? "Déterminé par les colonnes de A" : ""}
                      />
                      <StyledTextField
                        label="Colonnes"
                        type="number"
                        value={workflowData.matrixBDimensions[1]}
                        onChange={(e) => handleDimensionChange('matrixB', 1, e.target.value)}
                        InputProps={{ 
                          inputProps: { min: 2, max: 1000 },
                          readOnly: workflowData.type === 'MATRIX_ADDITION'
                        }}
                        disabled={workflowData.type === 'MATRIX_ADDITION'}
                        helperText={workflowData.type === 'MATRIX_ADDITION' ? "Égal aux colonnes de A" : ""}
                      />
                    </Stack>
                    
                    <Button 
                      variant="outlined" 
                      color="primary"
                      onClick={regenerateRandomMatrices}
                      sx={{ 
                        mt: 2,
                        borderRadius: 50,
                        borderColor: 'rgba(76, 175, 80, 0.5)',
                        color: '#4CAF50'
                      }}
                    >
                      Régénérer les matrices aléatoires
                    </Button>
                  </Stack>
                </Box>
              )}
              
              {workflowData.generationMode === 'upload' && (
                <Box>
                  <Typography variant="subtitle1" color="primary" gutterBottom>
                    Téléchargement des matrices
                  </Typography>
                  
                  <Stack spacing={3}>
                    <Box>
                      <Typography variant="body1" gutterBottom>Matrice A</Typography>
                      <input
                        id="file-upload-a"
                        type="file"
                        style={{ display: 'none' }}
                        onChange={(e) => handleFileChange('matrixA', e)}
                        accept=".csv,.json,.npy,.txt"
                      />
                      <label htmlFor="file-upload-b" style={{ width: '100%', display: 'block' }}>
                        <UploadButton component="span">
                          <Stack direction="column" alignItems="center" spacing={1}>
                            <CloudUpload sx={{ fontSize: 40, color: 'rgba(76, 175, 80, 0.8)' }} />
                            <Typography>
                              {selectedFileB ? selectedFileB.name : "Cliquez pour télécharger la matrice B"}
                            </Typography>
                          </Stack>
                        </UploadButton>
                      </label>
                    </Box>
                    
                    <Alert severity="info" sx={{ backgroundColor: 'rgba(33, 150, 243, 0.1)' }}>
                      Formats supportés: CSV, JSON, NPY et TXT. 
                      {workflowData.type === 'MATRIX_ADDITION' && 
                        "Pour l'addition, les deux matrices doivent avoir les mêmes dimensions."}
                      {workflowData.type === 'MATRIX_MULTIPLICATION' && 
                        "Pour la multiplication, le nombre de colonnes de A doit être égal au nombre de lignes de B."}
                    </Alert>
                  </Stack>
                </Box>
              )}
            </Box>
            
            <Divider sx={{ my: 3, borderColor: 'rgba(255, 255, 255, 0.1)' }} />
            
            <Typography variant="subtitle1" color="primary" gutterBottom>
              Paramètres de calcul
            </Typography>
            
            <Stack spacing={2} sx={{ mt: 2 }}>
              <FormControl fullWidth>
                <InputLabel id="algorithm-label">Algorithme</InputLabel>
                <StyledSelect
                  labelId="algorithm-label"
                  name="algorithm"
                  value={workflowData.algorithm}
                  onChange={handleChange}
                  label="Algorithme"
                >
                  {matrixAlgorithms.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.display}
                    </MenuItem>
                  ))}
                </StyledSelect>
              </FormControl>
              
              <FormControl fullWidth>
                <InputLabel id="precision-label">Précision</InputLabel>
                <StyledSelect
                  labelId="precision-label"
                  name="precision"
                  value={workflowData.precision}
                  onChange={handleChange}
                  label="Précision"
                >
                  {precisionTypes.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.display}
                    </MenuItem>
                  ))}
                </StyledSelect>
              </FormControl>
              
              <StyledTextField
                fullWidth
                label="Taille des blocs"
                name="blockSize"
                type="number"
                value={workflowData.blockSize}
                onChange={handleChange}
                InputProps={{ inputProps: { min: 10, max: 1000 } }}
                helperText="Pour l'algorithme par blocs - taille recommandée: 50-200"
              />
            </Stack>
          </Box>
        );
        
      case 3:
        return (
          <Box>
            <CardTitle>Configuration des volontaires</CardTitle>
            
            <Stack spacing={3}>
              <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                Définissez le nombre et les types de volontaires nécessaires pour exécuter ce workflow.
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 2 }}>
                <StyledTextField
                  label="Nombre min. de volontaires"
                  name="minVolunteers"
                  type="number"
                  value={workflowData.minVolunteers}
                  onChange={handleChange}
                  InputProps={{ inputProps: { min: 1, max: 100 } }}
                />
                
                <StyledTextField
                  label="Nombre max. de volontaires"
                  name="maxVolunteers"
                  type="number"
                  value={workflowData.maxVolunteers}
                  onChange={handleChange}
                  InputProps={{ inputProps: { min: 1, max: 1000 } }}
                />
              </Box>
              
              <Typography variant="subtitle1" color="primary" gutterBottom sx={{ mt: 2 }}>
                Préférences de types de volontaires
              </Typography>
              
              <Box sx={{ display: 'flex', mb: 2 }}>
                <FormControl fullWidth>
                  <InputLabel id="pref-type-label">Type de volontaire</InputLabel>
                  <StyledSelect
                    labelId="pref-type-label"
                    value={currentPreference}
                    onChange={(e) => setCurrentPreference(e.target.value)}
                    label="Type de volontaire"
                  >
                    {volunteerTypes.map(option => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.display}
                      </MenuItem>
                    ))}
                  </StyledSelect>
                </FormControl>
                <IconButton 
                  onClick={handleAddPreference} 
                  sx={{ ml: 1, color: '#4CAF50' }}
                >
                  <Add />
                </IconButton>
              </Box>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
                {workflowData.volunteerPreferences.length > 0 ? (
                  workflowData.volunteerPreferences.map(pref => (
                    <StyledChip
                      key={pref}
                      label={volunteerTypes.find(t => t.value === pref)?.display || pref}
                      onDelete={() => handleRemovePreference(pref)}
                      sx={{ m: 0.5 }}
                    />
                  ))
                ) : (
                  <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.5)', fontStyle: 'italic' }}>
                    Aucune préférence spécifiée (tous types acceptés)
                  </Typography>
                )}
              </Box>
              
              <Alert severity="info" sx={{ backgroundColor: 'rgba(33, 150, 243, 0.1)', mt: 2 }}>
                <Typography variant="body2">
                  Recommandations basées sur votre configuration:
                </Typography>
                <ul style={{ marginTop: 8, marginBottom: 0 }}>
                  {workflowData.matrixADimensions[0] * workflowData.matrixADimensions[1] > 10000 && (
                    <li>Pour des matrices de cette taille, privilégiez les volontaires avec "Mémoire importante"</li>
                  )}
                  {workflowData.type === 'MATRIX_MULTIPLICATION' && (
                    <li>Pour la multiplication de matrices, "Calcul intensif CPU" est recommandé</li>
                  )}
                  {workflowData.algorithm === 'strassen' && (
                    <li>L'algorithme de Strassen est optimisé pour les GPU</li>
                  )}
                </ul>
              </Alert>
            </Stack>
          </Box>
        );
        
      case 4:
        return (
          <Box>
            <CardTitle>Récapitulatif du workflow</CardTitle>
            
            <Paper sx={{ p: 3, mb: 3, background: 'rgba(255, 255, 255, 0.05)', borderRadius: 3 }}>
              <Stack spacing={2}>
                <Box>
                  <Typography variant="body2" color="textSecondary">Nom</Typography>
                  <Typography variant="body1" sx={{ fontSize: '1.1rem', fontWeight: 500 }}>
                    {workflowData.name || "(Non spécifié)"}
                  </Typography>
                </Box>
                
                <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                
                <Box>
                  <Typography variant="body2" color="textSecondary">Type d'opération</Typography>
                  <Typography variant="body1" sx={{ fontSize: '1.1rem', fontWeight: 500 }}>
                    {workflowData.displayType || workflowTypes.find(t => t.value === workflowData.type)?.display}
                  </Typography>
                </Box>
                
                <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                
                <Box>
                  <Typography variant="body2" color="textSecondary">Matrices</Typography>
                  <Typography variant="body1">
                    Matrice A: {workflowData.matrixADimensions[0]} × {workflowData.matrixADimensions[1]}
                    {selectedFileA && ` (fichier: ${selectedFileA.name})`}
                  </Typography>
                  <Typography variant="body1">
                    Matrice B: {workflowData.matrixBDimensions[0]} × {workflowData.matrixBDimensions[1]}
                    {selectedFileB && ` (fichier: ${selectedFileB.name})`}
                  </Typography>
                </Box>
                
                <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                
                <Box>
                  <Typography variant="body2" color="textSecondary">Paramètres de calcul</Typography>
                  <Typography variant="body1">
                    Algorithme: {matrixAlgorithms.find(a => a.value === workflowData.algorithm)?.display}
                  </Typography>
                  <Typography variant="body1">
                    Précision: {precisionTypes.find(p => p.value === workflowData.precision)?.display}
                  </Typography>
                  <Typography variant="body1">
                    Taille des blocs: {workflowData.blockSize}
                  </Typography>
                </Box>
                
                <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                
                <Box>
                  <Typography variant="body2" color="textSecondary">Configuration des volontaires</Typography>
                  <Typography variant="body1">
                    Min: {workflowData.minVolunteers}, Max: {workflowData.maxVolunteers}
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>Préférences:</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', mt: 0.5 }}>
                    {workflowData.volunteerPreferences.length > 0 ? (
                      workflowData.volunteerPreferences.map(pref => (
                        <StyledChip 
                          key={pref} 
                          label={volunteerTypes.find(t => t.value === pref)?.display || pref}
                          size="small"
                          sx={{ mr: 0.5, mb: 0.5 }}
                        />
                      ))
                    ) : (
                      <Typography variant="body2">(Tous types acceptés)</Typography>
                    )}
                  </Box>
                </Box>
                
                <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                
                <Box>
                  <Typography variant="body2" color="textSecondary">Description</Typography>
                  <Typography variant="body1">
                    {workflowData.description || "(Aucune description)"}
                  </Typography>
                </Box>
                
                <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                
                <Box>
                  <Typography variant="body2" color="textSecondary">Tags</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', mt: 1 }}>
                    {workflowData.tags.length > 0 ? (
                      workflowData.tags.map(tag => (
                        <StyledChip key={tag} label={tag} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                      ))
                    ) : (
                      <Typography variant="body2">(Aucun tag)</Typography>
                    )}
                  </Box>
                </Box>
              </Stack>
            </Paper>
            
            <Typography sx={{ textAlign: 'center', color: 'rgba(255,255,255,0.7)', mb: 3 }}>
              En soumettant ce workflow, le système va découper et distribuer le travail aux volontaires 
              disponibles selon les critères spécifiés. Vous pourrez suivre la progression en temps réel.
            </Typography>
          </Box>
        );
        
      default:
        return null;
    }
  };

  return (
    <StyledContainer maxWidth="md">
      <PageTitle>Nouveau Workflow Matriciel</PageTitle>
      
      <FormCard>
        <CardContent sx={{ p: 4 }}>
          {renderStepContent()}
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
            {currentStep > 1 && (
              <Button
                variant="outlined"
                onClick={goToPreviousStep}
                sx={{ 
                  borderRadius: 50,
                  borderColor: 'rgba(255, 255, 255, 0.2)',
                  color: 'rgba(255, 255, 255, 0.7)',
                  '&:hover': {
                    borderColor: 'rgba(255, 255, 255, 0.5)',
                    backgroundColor: 'rgba(255, 255, 255, 0.05)'
                  }
                }}
              >
                Retour
              </Button>
            )}
            
            <Box sx={{ ml: 'auto' }}>
              {currentStep < 4 ? (
                <StyledButton
                  variant="contained"
                  onClick={goToNextStep}
                  endIcon={<ArrowForward />}
                >
                  Continuer
                </StyledButton>
              ) : (
                <StyledButton
                  variant="contained"
                  onClick={handleSubmit}
                  disabled={loading}
                >
                  {loading ? "Soumission..." : "Soumettre le workflow"}
                </StyledButton>
              )}
            </Box>
          </Box>
        </CardContent>
      </FormCard>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {[1, 2, 3, 4].map(step => (
            <React.Fragment key={step}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: currentStep >= step ? '#4CAF50' : 'rgba(255, 255, 255, 0.2)',
                  transition: 'all 0.3s',
                  cursor: 'pointer',
                  boxShadow: currentStep === step ? '0 0 0 3px rgba(76, 175, 80, 0.2)' : 'none'
                }}
                onClick={() => setCurrentStep(step)}
              />
              {step < 4 && (
                <Box
                  sx={{
                    width: 30,
                    height: 2,
                    backgroundColor: currentStep > step ? '#4CAF50' : 'rgba(255, 255, 255, 0.2)',
                    transition: 'all 0.3s'
                  }}
                />
              )}
            </React.Fragment>
          ))}
        </Box>
      </Box>
      
      {/* Message d'erreur */}
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setError(null)} 
          severity="error" 
          sx={{ width: '100%', bgcolor: 'rgba(211, 47, 47, 0.9)' }}
        >
          {error}
        </Alert>
      </Snackbar>
    </StyledContainer>
  );
};

export default CreateMatrixWorkflow;