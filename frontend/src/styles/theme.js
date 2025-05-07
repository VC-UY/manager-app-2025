// src/styles/theme.js
import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    primary: {
      main: '#0A2463', // Bleu nuit
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#4CAF50', // Vert
      contrastText: '#FFFFFF',
    },
    error: {
      main: '#D90429', // Rouge
    },
    background: {
      default: '#F8F9FA',
      paper: '#FFFFFF',
    }
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
        containedPrimary: {
          boxShadow: '0 4px 10px rgba(10, 36, 99, 0.2)',
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
        }
      }
    }
  }
});