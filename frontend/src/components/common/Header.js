import React, { useState, useEffect } from 'react';
import { 
  AppBar, Toolbar, Box, Button, Typography, IconButton, 
  Menu, MenuItem, Badge, Tooltip, Divider
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AccountCircle, Notifications, Dashboard, PlayArrow,
  Assessment, HelpOutline, CloudDone, CloudOff, Refresh, Group
} from '@mui/icons-material';
import coordinationService from '../../services/CoordinationService';

const StyledAppBar = styled(AppBar)(({ theme }) => ({
  background: 'rgba(10, 36, 99, 0.85)',
  backdropFilter: 'blur(10px)',
  boxShadow: '0 4px 30px rgba(0, 0, 0, 0.1)',
  borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
}));

const NavButton = styled(Button)(({ theme, selected }) => ({
  color: 'white',
  margin: theme.spacing(0, 0.5),
  padding: theme.spacing(1, 2),
  borderRadius: 8,
  position: 'relative',
  overflow: 'hidden',
  '&::after': selected ? {
    content: '""',
    position: 'absolute',
    bottom: 0,
    left: '50%',
    transform: 'translateX(-50%)',
    width: '30%',
    height: 3,
    backgroundColor: theme.palette.secondary.main,
    borderRadius: '3px 3px 0 0',
  } : {}
}));

const Logo = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginRight: theme.spacing(3),
  '& img': {
    height: 30,
    marginRight: theme.spacing(1),
  },
  '& .text': {
    fontWeight: 700,
    background: 'linear-gradient(45deg, #FFFFFF 30%, #4CAF50 90%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  }
}));

const Header = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [anchorEl, setAnchorEl] = useState(null);
  const [notificationAnchorEl, setNotificationAnchorEl] = useState(null);
  const [coordinatorAnchorEl, setCoordinatorAnchorEl] = useState(null);
  const [coordinatorStatus, setCoordinatorStatus] = useState({
    connected: false,
    authenticated: false
  });
  const [volunteers, setVolunteers] = useState([]);

  useEffect(() => {
    // S'abonner au statut du coordinateur
    const unsubscribeCoordinator = coordinationService.subscribeToCoordinatorStatus(
      (status) => setCoordinatorStatus(status)
    );
    
    // S'abonner aux mises à jour des volontaires
    const unsubscribeVolunteers = coordinationService.subscribeToVolunteers(
      (newVolunteers) => setVolunteers(newVolunteers)
    );
    
    // Nettoyer les abonnements
    return () => {
      unsubscribeCoordinator();
      unsubscribeVolunteers();
    };
  }, []);

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleNotificationOpen = (event) => {
    setNotificationAnchorEl(event.currentTarget);
  };

  const handleNotificationClose = () => {
    setNotificationAnchorEl(null);
  };
  
  const handleCoordinatorMenuOpen = (event) => {
    setCoordinatorAnchorEl(event.currentTarget);
  };

  const handleCoordinatorMenuClose = () => {
    setCoordinatorAnchorEl(null);
  };
  
  const handleAuthentication = async () => {
    handleCoordinatorMenuClose();
    await coordinationService.authenticateWithCoordinator();
  };
  
  const handleRefreshVolunteers = async () => {
    handleCoordinatorMenuClose();
    await coordinationService.refreshVolunteers();
  };

  const navItems = [
    { label: 'Accueil', path: '/', icon: <Dashboard /> },
    { label: 'Créer', path: '/create', icon: <PlayArrow /> },
    { label: 'Analyses', path: '/analytics', icon: <Assessment /> },
    { label: 'Aide', path: '/help', icon: <HelpOutline /> },
  ];

  return (
    <StyledAppBar position="sticky">
      <Toolbar>
        <Logo onClick={() => navigate('/')} sx={{ cursor: 'pointer' }}>
          {/* Vous pouvez remplacer ceci par votre logo réel */}
          <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDQwIDQwIiBmaWxsPSJub25lIj4KPHBhdGggZD0iTTIwIDAuODMzMzM0QzkuMjkxNjcgMC44MzMzMzQgMC44MzMzMzQgOS4yOTE2NyAwLjgzMzMzNCAyMEMwLjgzMzMzNCAzMC43MDgzIDkuMjkxNjcgMzkuMTY2NyAyMCAzOS4xNjY3QzMwLjcwODMgMzkuMTY2NyAzOS4xNjY3IDMwLjcwODMgMzkuMTY2NyAyMEMzOS4xNjY3IDkuMjkxNjcgMzAuNzA4MyAwLjgzMzMzNCAyMCAwLjgzMzMzNFoiIGZpbGw9IiMwQTI0NjMiLz4KPHBhdGggZD0iTTIwIDkuMTY2NjdDMTQuMTA0MiA5LjE2NjY3IDkuMTY2NjcgMTQuMTA0MiA5LjE2NjY3IDIwQzkuMTY2NjcgMjUuODk1OCAxNC4xMDQyIDMwLjgzMzMgMjAgMzAuODMzM0MyNS44OTU4IDMwLjgzMzMgMzAuODMzMyAyNS44OTU4IDMwLjgzMzMgMjBDMzAuODMzMyAxNC4xMDQyIDI1Ljg5NTggOS4xNjY2NyAyMCA5LjE2NjY3WiIgZmlsbD0iIzRDQUY1MCIvPgo8cGF0aCBkPSJNMjAgMTUuODMzM0MxNy43NzkyIDE1LjgzMzMgMTUuODMzMyAxNy43NzkyIDE1LjgzMzMgMjBDMTUuODMzMyAyMi4yMjA4IDE3Ljc3OTIgMjQuMTY2NyAyMCAyNC4xNjY3QzIyLjIyMDggMjQuMTY2NyAyNC4xNjY3IDIyLjIyMDggMjQuMTY2NyAyMEMyNC4xNjY3IDE3Ljc3OTIgMjIuMjIwOCAxNS44MzMzIDIwIDE1LjgzMzNaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4=" alt="Logo" />
          <Typography variant="h6" className="text">Workflow Manager</Typography>
        </Logo>

        <Box sx={{ flexGrow: 1, display: 'flex' }}>
          {navItems.map((item) => (
            <NavButton
              key={item.path}
              startIcon={item.icon}
              onClick={() => navigate(item.path)}
              selected={location.pathname === item.path}
            >
              {item.label}
            </NavButton>
          ))}
        </Box>

        <Box sx={{ flexGrow: 0 }}>
          {/* Bouton Statut du Coordinateur */}
          <Tooltip title="Statut du Coordinateur">
            <IconButton
              size="large"
              color="inherit"
              onClick={handleCoordinatorMenuOpen}
            >
              {coordinatorStatus.authenticated ? (
                <Badge badgeContent={volunteers.length} color="secondary">
                  <CloudDone />
                </Badge>
              ) : (
                <CloudOff />
              )}
            </IconButton>
          </Tooltip>
          
          {/* Menu du Coordinateur */}
          <Menu
            anchorEl={coordinatorAnchorEl}
            open={Boolean(coordinatorAnchorEl)}
            onClose={handleCoordinatorMenuClose}
            PaperProps={{
              sx: {
                backgroundColor: 'rgba(14, 28, 54, 0.95)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
              }
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <Box sx={{ px: 2, py: 1 }}>
              <Typography variant="subtitle1">Statut du Coordinateur</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <Badge
                  color={coordinatorStatus.authenticated ? "success" : "error"}
                  variant="dot"
                  sx={{ mr: 1 }}
                >
                  <CloudDone />
                </Badge>
                <Typography variant="body2">
                  {coordinatorStatus.authenticated ? "Authentifié" : "Non authentifié"}
                </Typography>
              </Box>
            </Box>
            
            <Divider sx={{ my: 1, backgroundColor: 'rgba(255, 255, 255, 0.1)' }} />
            
            <Box sx={{ px: 2, py: 1 }}>
              <Typography variant="subtitle1">Volontaires</Typography>
              <Typography variant="body2">
                {volunteers.length} volontaires disponibles
              </Typography>
            </Box>
            
            <Divider sx={{ my: 1, backgroundColor: 'rgba(255, 255, 255, 0.1)' }} />
            
            <MenuItem onClick={handleAuthentication}>
              <CloudDone sx={{ mr: 1 }} />
              Authentifier
            </MenuItem>
            
            <MenuItem onClick={handleRefreshVolunteers}>
              <Refresh sx={{ mr: 1 }} />
              Rafraîchir les volontaires
            </MenuItem>
          </Menu>
          
          {/* Notifications */}
          <Tooltip title="Notifications">
            <IconButton
              size="large"
              color="inherit"
              onClick={handleNotificationOpen}
            >
              <Badge badgeContent={4} color="error">
                <Notifications />
              </Badge>
            </IconButton>
          </Tooltip>
          
          {/* Menu des notifications */}
          <Menu
            anchorEl={notificationAnchorEl}
            open={Boolean(notificationAnchorEl)}
            onClose={handleNotificationClose}
            PaperProps={{
              sx: {
                backgroundColor: 'rgba(14, 28, 54, 0.95)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
                width: 320,
              }
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <Box sx={{ px: 2, py: 1 }}>
              <Typography variant="subtitle1">Notifications</Typography>
            </Box>
            
            <Divider sx={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }} />
            
            <MenuItem onClick={() => { navigate('/status/abc123'); handleNotificationClose(); }}>
              <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                <Typography variant="body2" fontWeight="bold">
                  Workflow "Analyse de logs" terminé
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Il y a 5 minutes
                </Typography>
              </Box>
            </MenuItem>
            
            <MenuItem onClick={() => { navigate('/status/def456'); handleNotificationClose(); }}>
              <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                <Typography variant="body2" fontWeight="bold">
                  Erreur dans le workflow "Calcul de hachage"
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Il y a 12 minutes
                </Typography>
              </Box>
            </MenuItem>
            
            <MenuItem onClick={handleNotificationClose}>
              <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                <Typography variant="body2" fontWeight="bold">
                  3 nouveaux volontaires disponibles
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Il y a 27 minutes
                </Typography>
              </Box>
            </MenuItem>
            
            <MenuItem onClick={handleNotificationClose}>
              <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                <Typography variant="body2" fontWeight="bold">
                  Coordinateur connecté
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Il y a 35 minutes
                </Typography>
              </Box>
            </MenuItem>
            
            <Divider sx={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }} />
            
            <MenuItem onClick={handleNotificationClose} sx={{ justifyContent: 'center' }}>
              <Typography variant="body2" color="primary">
                Voir toutes les notifications
              </Typography>
            </MenuItem>
          </Menu>

          {/* Profil Utilisateur */}
          <Tooltip title="Profil utilisateur">
            <IconButton
              size="large"
              edge="end"
              color="inherit"
              onClick={handleMenuOpen}
            >
              <AccountCircle />
            </IconButton>
          </Tooltip>
          
          {/* Menu de profil */}
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            PaperProps={{
              sx: {
                backgroundColor: 'rgba(14, 28, 54, 0.95)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
              }
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <MenuItem onClick={handleMenuClose}>Profil</MenuItem>
            <MenuItem onClick={handleMenuClose}>Paramètres</MenuItem>
            <Divider sx={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }} />
            <MenuItem onClick={handleMenuClose}>Déconnexion</MenuItem>
          </Menu>
        </Box>
      </Toolbar>
    </StyledAppBar>
  );
};

export default Header;