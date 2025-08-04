import { useEffect, useRef, useState, useCallback } from 'react';
import { getWebSocketManager } from '../lib/websocket';

interface UseWebSocketOptions {
  onWorkflowUpdate?: (data: any) => void;
  onWorkflowStatusChange?: (data: any) => void;
  onTaskUpdate?: (data: any) => void;
  onTaskProgress?: (data: any) => void;
  onVolunteerUpdate?: (data: any) => void;
  onVolunteerStatus?: (data: any) => void;
  onCustomEvent?: (type: string, data: any) => void;
  autoConnect?: boolean;
}

export const useWebSocket = (options: UseWebSocketOptions = {}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const wsManager = useRef(getWebSocketManager());
  const { autoConnect = true } = options;

  // Fonction pour se connecter
  const connect = useCallback(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      setConnectionError('Token d\'authentification manquant');
      return;
    }

    setConnectionError(null);
    wsManager.current.connect(token, {
      onConnect: () => {
        console.log('WebSocket connecté via hook');
        setIsConnected(true);
        setConnectionError(null);
      },
      onDisconnect: () => {
        console.log('WebSocket déconnecté via hook');
        setIsConnected(false);
      },
      onError: (error) => {
        console.error('Erreur WebSocket via hook:', error);
        setConnectionError('Erreur de connexion WebSocket');
        setIsConnected(false);
      },
      onWorkflowUpdate: (data) => {
        console.log('Mise à jour workflow reçue:', data);
        setLastMessage({ type: 'workflow_update', data, timestamp: Date.now() });
        options.onWorkflowUpdate?.(data);
      },
      onWorkflowStatusChange: (data) => {
        console.log('Changement statut workflow reçu:', data);
        setLastMessage({ type: 'workflow_status_change', data, timestamp: Date.now() });
        options.onWorkflowStatusChange?.(data);
      },
      onTaskUpdate: (data) => {
        console.log('Mise à jour tâche reçue:', data);
        setLastMessage({ type: 'task_update', data, timestamp: Date.now() });
        options.onTaskUpdate?.(data);
      },
      onTaskProgress: (data) => {
        console.log('Progression tâche reçue:', data);
        if (data.status === 'complete') {
          data.progress = 100;
        }
        setLastMessage({ type: 'task_progress', data, timestamp: Date.now() });
        options.onTaskProgress?.(data);
      },
      onVolunteerUpdate: (data) => {
        console.log('Mise à jour volontaire reçue:', data);
        setLastMessage({ type: 'volunteer_update', data, timestamp: Date.now() });
        options.onVolunteerUpdate?.(data);
      },
      onVolunteerStatus: (data) => {
        console.log('Statut volontaire reçu:', data);
        setLastMessage({ type: 'volunteer_status', data, timestamp: Date.now() });
        options.onVolunteerStatus?.(data);
      },
      onCustomEvent: (type, data) => {
        console.log('Événement personnalisé reçu:', type, data);
        setLastMessage({ type, data, timestamp: Date.now() });
        options.onCustomEvent?.(type, data);
      },
    });
  }, [options]);

  // Fonction pour se déconnecter
  const disconnect = useCallback(() => {
    wsManager.current.disconnect();
    setIsConnected(false);
  }, []);

  // Fonction pour forcer la reconnexion
  const forceReconnect = useCallback(() => {
    wsManager.current.forceReconnect();
  }, []);

  // Fonctions d'abonnement
  const subscribeToWorkflow = useCallback((workflowId: string) => {
    if (isConnected) {
      wsManager.current.subscribeToWorkflow(workflowId);
    } else {
      console.warn('WebSocket non connecté, impossible de s\'abonner au workflow');
    }
  }, [isConnected]);

  const subscribeToTask = useCallback((taskId: string) => {
    if (isConnected) {
      wsManager.current.subscribeToTask(taskId);
    } else {
      console.warn('WebSocket non connecté, impossible de s\'abonner à la tâche');
    }
  }, [isConnected]);

  const subscribeToVolunteer = useCallback((volunteerId: string) => {
    if (isConnected) {
      wsManager.current.subscribeToVolunteer(volunteerId);
    } else {
      console.warn('WebSocket non connecté, impossible de s\'abonner au volontaire');
    }
  }, [isConnected]);

  // Fonction pour envoyer un ping
  const ping = useCallback(() => {
    if (isConnected) {
      wsManager.current.ping();
    }
  }, [isConnected]);

  // Obtenir l'état de la connexion
  const getConnectionState = useCallback(() => {
    const readyState = wsManager.current.getReadyState();
    switch (readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'open';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'closed';
      default:
        return 'unknown';
    }
  }, []);

  // Connexion automatique
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    // Nettoyage lors du démontage du composant
    return () => {
      if (!autoConnect) {
        disconnect();
      }
    };
  }, [autoConnect, connect, disconnect]);

  // Vérification périodique de la connexion
  useEffect(() => {
    if (autoConnect) {
      const checkConnection = setInterval(() => {
        const actuallyConnected = wsManager.current.isConnected();
        if (actuallyConnected !== isConnected) {
          setIsConnected(actuallyConnected);
        }
      }, 5000); // Vérifier toutes les 5 secondes

      return () => clearInterval(checkConnection);
    }
  }, [autoConnect, isConnected]);

  return {
    isConnected,
    connectionError,
    lastMessage,
    connect,
    disconnect,
    forceReconnect,
    subscribeToWorkflow,
    subscribeToTask,
    subscribeToVolunteer,
    ping,
    getConnectionState,
  };
};