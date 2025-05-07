// src/pages/Help.js
import React from 'react';
import { Box, Typography, Card, Grid, Accordion, AccordionSummary, AccordionDetails, Button } from '@mui/material';
import { styled } from '@mui/material/styles';
import { ExpandMore, PlayArrow, School, ContactSupport, Build, LiveHelp } from '@mui/icons-material';

const GlowingCard = styled(Card)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
  position: 'relative',
  overflow: 'hidden',
  '&:before': {
    content: '""',
    position: 'absolute',
    top: -20,
    left: -20,
    right: -20,
    bottom: -20,
    background: 'radial-gradient(circle at center, rgba(76, 175, 80, 0.15) 0%, rgba(10, 36, 99, 0) 70%)',
    zIndex: 0,
  }
}));

const StyledAccordion = styled(Accordion)(({ theme }) => ({
  background: 'rgba(14, 28, 54, 0.6)',
  backdropFilter: 'blur(10px)',
  borderRadius: '12px !important',
  marginBottom: theme.spacing(2),
  color: 'white',
  '&:before': {
    display: 'none',
  },
  '& .MuiAccordionSummary-root': {
    borderRadius: 12,
  }
}));

const Help = () => {
  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Typography 
        variant="h3" 
        sx={{ 
          mb: 4, 
          fontWeight: 'bold',
          background: 'linear-gradient(45deg, #FFFFFF 30%, #4CAF50 90%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          textAlign: 'center'
        }}
      >
        Aide & Support
      </Typography>

      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <GlowingCard>
            <Typography variant="h5" gutterBottom>
              <School sx={{ mr: 1, verticalAlign: 'middle' }} />
              Guide de démarrage
            </Typography>
            
            <StyledAccordion>
              <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
                <Typography>Qu'est-ce que le calcul distribué volontaire ?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography>
                  Le calcul distribué volontaire est une approche qui permet d'utiliser la puissance de calcul inutilisée de nombreux ordinateurs 
                  volontaires pour résoudre des problèmes complexes. Chaque volontaire met à disposition une partie des ressources de son ordinateur 
                  (CPU, mémoire, stockage) pour exécuter des sous-tâches d'un workflow plus large.
                </Typography>
              </AccordionDetails>
            </StyledAccordion>
            
            <StyledAccordion>
              <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
                <Typography>Comment soumettre mon premier workflow ?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography paragraph>
                  Pour soumettre un workflow, suivez ces étapes simples :
                </Typography>
                <ol>
                  <li>Cliquez sur "Nouveau Workflow" dans la barre de navigation</li>
                  <li>Remplissez les informations générales (nom, description, type)</li>
                  <li>Spécifiez les ressources requises pour chaque sous-tâche</li>
                  <li>Configurez les paramètres techniques (image Docker, commande)</li>
                  <li>Vérifiez le récapitulatif et soumettez votre workflow</li>
                </ol>
                <Button 
                  variant="contained" 
                  color="secondary" 
                  startIcon={<PlayArrow />}
                  href="/create"
                  sx={{ mt: 2 }}
                >
                  Créer un workflow
                </Button>
              </AccordionDetails>
            </StyledAccordion>
            
            <StyledAccordion>
              <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
                <Typography>Comment suivre l'exécution de mon workflow ?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography>
                  Une fois votre workflow soumis, vous pouvez suivre son avancement en temps réel depuis la page de détail du workflow. 
                  Vous y verrez l'état global, la progression des sous-tâches, les volontaires impliqués, et les ressources utilisées. 
                  Le système vous alertera automatiquement en cas de problème lors de l'exécution.
                </Typography>
              </AccordionDetails>
            </StyledAccordion>
            
            <StyledAccordion>
              <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
                <Typography>Comment récupérer mes résultats ?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography>
                  Les résultats sont automatiquement agrégés une fois le workflow terminé. Vous pouvez les visualiser directement 
                  dans l'interface sous forme de graphiques, tableaux ou résumés textuels. Il est également possible d'exporter 
                  les résultats dans différents formats (JSON, CSV, PDF) pour une analyse plus approfondie avec vos outils habituels.
                </Typography>
              </AccordionDetails>
            </StyledAccordion>
          </GlowingCard>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <GlowingCard>
            <Typography variant="h5" gutterBottom>
              <LiveHelp sx={{ mr: 1, verticalAlign: 'middle' }} />
              Questions fréquentes
            </Typography>
            
            <StyledAccordion>
              <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
                <Typography>Mes données sont-elles sécurisées ?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography>
                  Oui, la sécurité est une priorité. Les données sont chiffrées pendant le transfert et le stockage. 
                  Les volontaires n'ont accès qu'aux données strictement nécessaires pour accomplir leur sous-tâche, 
                  et des mécanismes de vérification garantissent l'intégrité des résultats. Vous pouvez également 
                  spécifier des contraintes supplémentaires concernant la distribution de vos tâches.
                </Typography>
              </AccordionDetails>
            </StyledAccordion>
            
            <StyledAccordion>
              <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
                <Typography>Quels types de workflows sont supportés ?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography>
                  Notre système supporte quatre types principaux de workflows :
                </Typography>
                <ul>
                  <li><strong>Traitement de données</strong> : analyse de fichiers volumineux, transformations, ETL</li>
                  <li><strong>Calcul scientifique</strong> : simulations, modélisation, analyses mathématiques</li>
                  <li><strong>Rendu graphique</strong> : génération d'images, animations 3D, visualisations</li>
                  <li><strong>Apprentissage automatique</strong> : entraînement et inférence de modèles ML</li>
                </ul>
                <Typography sx={{ mt: 1 }}>
                  Chaque type est optimisé pour maximiser l'efficacité du découpage et de la distribution des tâches.
                </Typography>
              </AccordionDetails>
            </StyledAccordion>
            
            <StyledAccordion>
              <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
                <Typography>Comment fonctionne le découpage automatique ?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography>
                  Le découpage automatique analyse votre workflow pour identifier les parties qui peuvent être exécutées 
                  en parallèle. Il divise ensuite la tâche principale en sous-tâches indépendantes, en tenant compte des 
                  dépendances de données et de contrôle. L'algorithme optimise le découpage en fonction du type de workflow 
                  et des ressources disponibles pour maximiser le parallélisme tout en minimisant les transferts de données.
                </Typography>
              </AccordionDetails>
            </StyledAccordion>
            
            <StyledAccordion>
              <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
                <Typography>Que faire en cas d'échec d'une sous-tâche ?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography>
                  En cas d'échec d'une sous-tâche, le système tente automatiquement de la réassigner à un autre volontaire 
                  jusqu'à 3 fois. Si l'échec persiste, vous êtes alerté et pouvez choisir entre : attendre qu'un volontaire 
                  plus adapté soit disponible, ajuster les paramètres de la sous-tâche, ou ignorer cette sous-tâche si elle 
                  n'est pas critique pour le résultat global.
                </Typography>
              </AccordionDetails>
            </StyledAccordion>
          </GlowingCard>
          
          <GlowingCard>
            <Typography variant="h5" gutterBottom>
              <ContactSupport sx={{ mr: 1, verticalAlign: 'middle' }} />
              Besoin d'aide supplémentaire ?
            </Typography>
            
            <Typography paragraph>
              Notre équipe de support est disponible pour vous aider avec toute question technique ou problème 
              que vous pourriez rencontrer.
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Button 
                  fullWidth
                  variant="outlined" 
                  color="primary"
                  startIcon={<Build />}
                  sx={{ 
                    py: 1.5,
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    color: 'white',
                    '&:hover': {
                      borderColor: 'rgba(255, 255, 255, 0.4)',
                      backgroundColor: 'rgba(255, 255, 255, 0.05)'
                    }
                  }}
                >
                  Documentation
                </Button>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Button 
                  fullWidth
                  variant="contained" 
                  color="secondary"
                  sx={{ py: 1.5 }}
                >
                  Contacter le support
                </Button>
              </Grid>
            </Grid>
          </GlowingCard>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Help;