// src/pages/Results.js
import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box, Typography, Card, Button, Tabs, Tab, Paper,
  CircularProgress, Grid, Menu, MenuItem, ListItemIcon, ListItemText
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  CloudDownload, Share, Article, Code, TableChart,
  PieChart, BarChart as BarChartIcon, LineChart as LineChartIcon,
  Image, TextSnippet, FormatListBulleted
} from '@mui/icons-material';
import { BarChart, Bar, LineChart, Line, PieChart as PieChartRechart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import api from '../services/api';

const GlowingCard = styled(Card)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
  transition: 'all 0.3s ease',
}));

const COLORS = ['#0A2463', '#4CAF50', '#9C27B0', '#FF9800', '#D90429'];

const Results = () => {
  const { id } = useParams();
  const [results, setResults] = useState(null);
  const [workflow, setWorkflow] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);
  const [exportMenuAnchor, setExportMenuAnchor] = useState(null);

  useEffect(() => {
    const fetchWorkflowDetails = async () => {
      try {
        const response = await api.get(`/workflows/${id}/`);
        setWorkflow(response.data);
      } catch (error) {
        console.error('Error fetching workflow:', error);
      }
    };

    const fetchResults = async () => {
      try {
        const response = await api.get(`/workflows/${id}/results/`);
        setResults(response.data);
      } catch (error) {
        console.error('Error fetching results:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflowDetails();
    fetchResults();
  }, [id]);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleExportClick = (event) => {
    setExportMenuAnchor(event.currentTarget);
  };

  const handleExportClose = () => {
    setExportMenuAnchor(null);
  };

  // Données simulées pour les graphiques
  const chartData = [
    { name: 'Sous-tâche 1', durée: 12, valeur: 240 },
    { name: 'Sous-tâche 2', durée: 8, valeur: 150 },
    { name: 'Sous-tâche 3', durée: 20, valeur: 320 },
    { name: 'Sous-tâche 4', durée: 15, valeur: 280 },
    { name: 'Sous-tâche 5', durée: 7, valeur: 120 },
    { name: 'Sous-tâche 6', durée: 10, valeur: 190 },
    { name: 'Sous-tâche 7', durée: 18, valeur: 310 },
    { name: 'Sous-tâche 8', durée: 11, valeur: 230 },
  ];

  const pieData = [
    { name: 'Terminé avec succès', value: 6 },
    { name: 'Avec avertissements', value: 1 },
    { name: 'Échoué', value: 1 },
  ];

  if (loading || !workflow) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <CircularProgress color="secondary" />
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Box sx={{ mb: 4 }}>
        <Typography 
          variant="h3" 
          sx={{ 
            fontWeight: 'bold',
            background: 'linear-gradient(45deg, #FFFFFF 30%, #4CAF50 90%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Résultats
        </Typography>
        <Typography variant="h5" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
          {workflow.name}
        </Typography>
      </Box>

      <GlowingCard>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="subtitle1">Résumé</Typography>
            <Typography variant="body2" color="textSecondary">
              Workflow terminé le {new Date(workflow.completed_at || Date.now()).toLocaleString()}
            </Typography>
          </Box>
          <Box>
            <Button
              variant="contained"
              startIcon={<CloudDownload />}
              onClick={handleExportClick}
              sx={{ 
                background: 'linear-gradient(45deg, #0A2463 0%, #1A46A3 100%)',
                boxShadow: '0 4px 20px rgba(10, 36, 99, 0.4)',
              }}
            >
              Exporter
            </Button>
            <Menu
              anchorEl={exportMenuAnchor}
              open={Boolean(exportMenuAnchor)}
              onClose={handleExportClose}
              PaperProps={{
                sx: {
                  backgroundColor: 'rgba(14, 28, 54, 0.95)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: 2,
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                }
              }}
            >
              <MenuItem onClick={handleExportClose}>
                <ListItemIcon>
                  <Article sx={{ color: '#4CAF50' }} />
                </ListItemIcon>
                <ListItemText>Exporter en PDF</ListItemText>
              </MenuItem>
              <MenuItem onClick={handleExportClose}>
                <ListItemIcon>
                  <Code sx={{ color: '#2196F3' }} />
                </ListItemIcon>
                <ListItemText>Exporter en JSON</ListItemText>
              </MenuItem>
              <MenuItem onClick={handleExportClose}>
                <ListItemIcon>
                  <TableChart sx={{ color: '#9C27B0' }} />
                </ListItemIcon>
                <ListItemText>Exporter en CSV</ListItemText>
              </MenuItem>
              <MenuItem onClick={handleExportClose}>
                <ListItemIcon>
                  <Image sx={{ color: '#FF9800' }} />
                </ListItemIcon>
                <ListItemText>Exporter les graphiques</ListItemText>
              </MenuItem>
            </Menu>
          </Box>
        </Box>

        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, backgroundColor: 'rgba(14, 28, 54, 0.6)', borderRadius: 2, height: '100%' }}>
              <Typography variant="h5" gutterBottom>8</Typography>
              <Typography variant="body2" color="textSecondary">Sous-tâches exécutées</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, backgroundColor: 'rgba(14, 28, 54, 0.6)', borderRadius: 2, height: '100%' }}>
              <Typography variant="h5" gutterBottom>1h 42m</Typography>
              <Typography variant="body2" color="textSecondary">Temps d'exécution total</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, backgroundColor: 'rgba(14, 28, 54, 0.6)', borderRadius: 2, height: '100%' }}>
              <Typography variant="h5" gutterBottom>87.5%</Typography>
              <Typography variant="body2" color="textSecondary">Taux de réussite</Typography>
            </Paper>
          </Grid>
        </Grid>

        <Tabs 
          value={tabValue} 
          onChange={handleTabChange}
          sx={{ 
            borderBottom: 1, 
            borderColor: 'rgba(255, 255, 255, 0.1)',
            mb: 3 
          }}
        >
          <Tab icon={<BarChartIcon />} label="Graphiques" />
          <Tab icon={<TextSnippet />} label="Résumé textuel" />
          <Tab icon={<FormatListBulleted />} label="Résultats détaillés" />
        </Tabs>

        {tabValue === 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, backgroundColor: 'rgba(14, 28, 54, 0.6)', borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom>Durée d'exécution par sous-tâche</Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                      <XAxis dataKey="name" tick={{ fill: 'rgba(255, 255, 255, 0.7)' }} />
                      <YAxis tick={{ fill: 'rgba(255, 255, 255, 0.7)' }} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'rgba(14, 28, 54, 0.9)',
                          borderColor: 'rgba(255, 255, 255, 0.1)',
                          color: 'white'
                        }}
                      />
                      <Legend />
                      <Bar dataKey="durée" fill="#4CAF50" name="Durée (minutes)" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, backgroundColor: 'rgba(14, 28, 54, 0.6)', borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom>Valeur générée par sous-tâche</Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                      <XAxis dataKey="name" tick={{ fill: 'rgba(255, 255, 255, 0.7)' }} />
                      <YAxis tick={{ fill: 'rgba(255, 255, 255, 0.7)' }} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'rgba(14, 28, 54, 0.9)',
                          borderColor: 'rgba(255, 255, 255, 0.1)',
                          color: 'white'
                        }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="valeur" stroke="#2196F3" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>
            <Grid item xs={12}>
              <Paper sx={{ p: 3, backgroundColor: 'rgba(14, 28, 54, 0.6)', borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom>Répartition des statuts</Typography>
                <Box sx={{ height: 300, display: 'flex', justifyContent: 'center' }}>
                  <ResponsiveContainer width="50%" height="100%">
                    <PieChartRechart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={120}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'rgba(14, 28, 54, 0.9)',
                          borderColor: 'rgba(255, 255, 255, 0.1)',
                          color: 'white'
                        }}
                      />
                    </PieChartRechart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        )}

        {tabValue === 1 && (
          <Paper sx={{ p: 3, backgroundColor: 'rgba(14, 28, 54, 0.6)', borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom>Résumé de l'exécution</Typography>
            <Typography variant="body1" paragraph>
              Le workflow "{workflow.name}" a été exécuté avec succès sur le système de calcul distribué volontaire. Sur un total de 8 sous-tâches, 7 ont été complétées avec succès, tandis qu'une a rencontré des erreurs.
            </Typography>
            <Typography variant="body1" paragraph>
              La majorité des sous-tâches a été exécutée en moins de 15 minutes, avec une durée moyenne de 12.6 minutes par sous-tâche. La sous-tâche la plus longue a duré 20 minutes, tandis que la plus rapide n'a pris que 7 minutes.
            </Typography>
            <Typography variant="body1" paragraph>
              L'exécution a été répartie sur 3 volontaires différents, chacun prenant en charge entre 2 et 3 sous-tâches. Le taux d'utilisation moyen des ressources a été de 65% pour le CPU, 42% pour la mémoire et 28% pour le stockage.
            </Typography>
            <Typography variant="body1">
              Au total, le workflow a traité environ 2.3 Go de données et a généré 450 Mo de résultats.
            </Typography>
          </Paper>
        )}

        {tabValue === 2 && (
          <Paper sx={{ p: 3, backgroundColor: 'rgba(14, 28, 54, 0.6)', borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom>Résultats détaillés</Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Liste complète des résultats générés par chaque sous-tâche
            </Typography>
            
            {chartData.map((task, index) => (
              <Box 
                key={index} 
                sx={{ 
                  p: 2, 
                  mb: 2, 
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  borderRadius: 2,
                  borderLeft: `4px solid ${COLORS[index % COLORS.length]}`
                }}
              >
                <Typography variant="subtitle1">{task.name}</Typography>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Durée: {task.durée} minutes • Valeur: {task.valeur} unités
                </Typography>
                <Typography variant="body2">
                  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam in dui mauris. 
                  Vivamus hendrerit arcu sed erat molestie vehicula. Sed auctor neque eu tellus 
                  rhoncus ut eleifend nibh porttitor.
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                  <Button 
                    size="small" 
                    variant="outlined"
                    startIcon={<CloudDownload />}
                    sx={{ 
                      borderColor: 'rgba(255, 255, 255, 0.2)',
                      color: 'white',
                      '&:hover': {
                        borderColor: 'rgba(255, 255, 255, 0.4)',
                        backgroundColor: 'rgba(255, 255, 255, 0.05)'
                      }
                    }}
                  >
                    Télécharger
                  </Button>
                  <Button 
                    size="small" 
                    variant="outlined"
                    startIcon={<Share />}
                    sx={{ 
                      borderColor: 'rgba(255, 255, 255, 0.2)',
                      color: 'white',
                      '&:hover': {
                        borderColor: 'rgba(255, 255, 255, 0.4)',
                        backgroundColor: 'rgba(255, 255, 255, 0.05)'
                      }
                    }}
                  >
                    Partager
                  </Button>
                </Box>
              </Box>
            ))}
          </Paper>
        )}
      </GlowingCard>
    </Box>
  );
};

export default Results;