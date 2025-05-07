import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Backdrop, CircularProgress } from '@mui/material';
import ParticleBackground from './components/common/ParticleBackground';
import MathBackground from './components/common/MathBackground';
import Header from './components/common/Header';
import Home from './pages/Home';
import CreateWorkflow from './pages/CreateWorkflow';
import WorkflowStatus from './pages/WorkflowStatus';
import Results from './pages/Results';
import Help from './pages/Help';
import Analytics from './pages/Analytics';
import coordinationService from './services/CoordinationService';

// Création du thème
const theme = createTheme({
  palette: {
    primary: {
      main: '#0A2463',
    },
    secondary: {
      main: '#4CAF50',
    },
    error: {
      main: '#D90429',
    },
    background: {
      default: '#030C26',
      paper: 'rgba(14, 28, 54, 0.8)',
    },
    text: {
      primary: '#FFFFFF',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  typography: {
    fontFamily: '"Poppins", "Roboto", sans-serif',
    h1: {
      fontWeight: 700,
    },
    h2: {
      fontWeight: 600,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(135deg, rgba(14, 28, 54, 0.8) 0%, rgba(20, 40, 80, 0.8) 100%)',
          backdropFilter: 'blur(10px)',
          borderRadius: 16,
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '10px 24px',
        },
        containedPrimary: {
          background: 'linear-gradient(45deg, #0A2463 0%, #1A46A3 100%)',
          boxShadow: '0 4px 20px rgba(10, 36, 99, 0.4)',
          '&:hover': {
            background: 'linear-gradient(45deg, #0A2463 30%, #1A46A3 90%)',
            boxShadow: '0 6px 25px rgba(10, 36, 99, 0.6)',
            transform: 'translateY(-2px)',
          },
          transition: 'all 0.3s ease',
        },
        containedSecondary: {
          background: 'linear-gradient(45deg, #357a38 0%, #4CAF50 100%)',
          boxShadow: '0 4px 20px rgba(76, 175, 80, 0.4)',
          '&:hover': {
            background: 'linear-gradient(45deg, #357a38 30%, #4CAF50 90%)',
            boxShadow: '0 6px 25px rgba(76, 175, 80, 0.6)',
            transform: 'translateY(-2px)',
          },
          transition: 'all 0.3s ease',
        },
      },
    },
  },
});

function App() {
  // État pour suivre l'initialisation du service
  const [initializing, setInitializing] = useState(true);
  const [initError, setInitError] = useState(null);
  
  // Initialisation du service de coordination
  useEffect(() => {
    console.log("App.js - Démarrage de l'initialisation du service");
    
    // Fonction asynchrone pour initialiser le service
    const initializeService = async () => {
      try {
        // Initialiser le service avec un timeout si nécessaire
        const initPromise = coordinationService.initialize();
        
        // Timeout de 5 secondes pour éviter un blocage indéfini
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error("Délai d'initialisation du service dépassé")), 5000);
        });
        
        // Attendre soit l'initialisation réussie, soit le timeout
        await Promise.race([initPromise, timeoutPromise]);
        
        console.log("App.js - Service de coordination initialisé avec succès");
      } catch (error) {
        console.error("App.js - Erreur lors de l'initialisation du service de coordination:", error);
        setInitError(error.message);
        // Ne pas bloquer le rendu en cas d'erreur
      } finally {
        // Indiquer que la tentative d'initialisation est terminée
        setInitializing(false);
      }
    };
    
    // Démarrer l'initialisation
    initializeService();
    
    // Nettoyage à la déconnexion
    return () => {
      console.log("App.js - Déconnexion du service de coordination");
      try {
        coordinationService.disconnect();
      } catch (error) {
        console.error("App.js - Erreur lors de la déconnexion du service de coordination:", error);
      }
    };
  }, []);
  
  // Si l'initialisation est toujours en cours après 500ms, afficher un indicateur
  // Mais permettre le rendu de l'application si ça prend trop de temps
  useEffect(() => {
    if (initializing) {
      const timer = setTimeout(() => {
        console.log("App.js - L'initialisation prend du temps, continue le rendu");
        setInitializing(false);
      }, 1000);
      
      return () => clearTimeout(timer);
    }
  }, [initializing]);
  
  // Rendu de l'application
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      {/* Indicateur de chargement si l'initialisation est toujours en cours */}
      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={initializing}
      >
        <CircularProgress color="secondary" />
      </Backdrop>
      
      <div className="app-container" style={{ position: 'relative', minHeight: '100vh', overflow: 'hidden' }}>
        {/* Composants d'arrière-plan */}
        {!initializing && (
          <>
            <ParticleBackground />
            <MathBackground />
          </>
        )}
        
        {/* Contenu principal de l'application */}
        <BrowserRouter>
          <Header />
          <main style={{ padding: '24px', position: 'relative', zIndex: 1, minHeight: 'calc(100vh - 70px)' }}>
            {initError && (
              <div style={{ 
                background: 'rgba(217, 4, 41, 0.2)', 
                padding: '10px', 
                borderRadius: '8px',
                marginBottom: '20px',
                color: '#fff'
              }}>
                Attention: Certaines fonctionnalités peuvent être limitées - 
                Erreur d'initialisation du service: {initError}
              </div>
            )}
            
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/create" element={<CreateWorkflow />} />
              <Route path="/status/:id" element={<WorkflowStatus />} />
              <Route path="/results/:id" element={<Results />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/help" element={<Help />} />
            </Routes>
          </main>
        </BrowserRouter>
      </div>
    </ThemeProvider>
  );
}

export default App;