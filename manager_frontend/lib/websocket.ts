// lib/websocket.ts
interface WebSocketMessage {
    type: string;
    [key: string]: any;
  }
  
  interface WebSocketCallbacks {
    onWorkflowUpdate?: (data: any) => void;
    onWorkflowStatusChange?: (data: any) => void;
    onTaskUpdate?: (data: any) => void;
    onTaskProgress?: (data: any) => void;
    onVolunteerUpdate?: (data: any) => void;
    onVolunteerStatus?: (data: any) => void;
    onConnect?: () => void;
    onDisconnect?: () => void;
    onError?: (error: Event) => void;
    onCustomEvent?: (type: string, data: any) => void;
  }
  
  class WebSocketManager {
    private ws: WebSocket | null = null;
    private callbacks: WebSocketCallbacks = {};
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000;
    private token: string | null = null;
    private url: string;
    private isConnecting = false;
    private pingInterval: NodeJS.Timeout | null = null;
  
    constructor(url: string = 'ws://127.0.0.1:8000/ws/workflows/') {
      this.url = url;
    }
  
    connect(token: string, callbacks: WebSocketCallbacks = {}) {
      if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
        console.log('WebSocket déjà connecté ou en cours de connexion');
        return;
      }
  
      this.token = token;
      this.callbacks = callbacks;
      this.isConnecting = true;
      
      if (this.ws) {
        this.ws.close();
      }
  
      try {
        // Ajouter le token comme paramètre de requête
        const wsUrl = `${this.url}?token=${encodeURIComponent(token)}`;
        console.log('Connexion WebSocket à:', wsUrl);
        this.ws = new WebSocket(wsUrl);
  
        this.ws.onopen = (event) => {
          console.log('WebSocket connecté avec succès');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.callbacks.onConnect?.();
          
          // Démarrer le ping périodique
          this.startPingInterval();
        };
  
        this.ws.onmessage = (event) => {
          try {
            const data: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Erreur lors du parsing du message WebSocket:', error);
          }
        };
  
        this.ws.onclose = (event) => {
          console.log('WebSocket fermé:', event.code, event.reason);
          this.isConnecting = false;
          this.callbacks.onDisconnect?.();
          
          // Arrêter le ping
          this.stopPingInterval();
          
          // Tentative de reconnexion automatique
          if (this.reconnectAttempts < this.maxReconnectAttempts && event.code !== 1000) {
            this.reconnectAttempts++;
            console.log(`Tentative de reconnexion ${this.reconnectAttempts}/${this.maxReconnectAttempts} dans ${this.reconnectDelay * this.reconnectAttempts}ms`);
            
            setTimeout(() => {
              if (this.token) {
                this.connect(this.token, this.callbacks);
              }
            }, this.reconnectDelay * this.reconnectAttempts);
          }
        };
  
        this.ws.onerror = (error) => {
          console.error('Erreur WebSocket:', error);
          this.isConnecting = false;
          this.callbacks.onError?.(error);
        };
  
      } catch (error) {
        console.error('Erreur lors de la création de la connexion WebSocket:', error);
        this.isConnecting = false;
      }
    }
  
    private handleMessage(data: WebSocketMessage) {
      console.log('Message WebSocket reçu:', data);
  
      switch (data.type) {
        case 'connection_established':
          console.log('Connexion WebSocket confirmée:', data.message);
          break;
  
        case 'workflow_update':
          this.callbacks.onWorkflowUpdate?.(data);
          break;
  
        case 'workflow_status_change':
          this.callbacks.onWorkflowStatusChange?.(data);
          break;
  
        case 'task_update':
          this.callbacks.onTaskUpdate?.(data);
          break;
  
        case 'task_progress':
          this.callbacks.onTaskProgress?.(data);
          break;
  
        case 'volunteer_update':
          this.callbacks.onVolunteerUpdate?.(data);
          break;
  
        case 'volunteer_status':
          this.callbacks.onVolunteerStatus?.(data);
          break;
  
        case 'subscription_confirmed':
          console.log(`Abonnement confirmé pour ${data.subject} ${data.id}`);
          break;
  
        case 'pong':
          // Réponse au ping - pas besoin de traitement spécial
          break;
  
        default:
          console.log('Type de message non géré:', data.type);
          // Appeler le callback générique pour les événements personnalisés
          this.callbacks.onCustomEvent?.(data.type, data);
      }
    }
  
    // Méthodes pour s'abonner à des entités spécifiques
    subscribeToWorkflow(workflowId: string) {
      this.send({
        type: 'subscribe_workflow',
        workflow_id: workflowId
      });
    }
  
    subscribeToTask(taskId: string) {
      this.send({
        type: 'subscribe_task',
        task_id: taskId
      });
    }
  
    subscribeToVolunteer(volunteerId: string) {
      this.send({
        type: 'subscribe_volunteer',
        volunteer_id: volunteerId
      });
    }
  
    // Envoyer un ping pour maintenir la connexion
    ping() {
      this.send({ type: 'ping' });
    }
  
    private startPingInterval() {
      this.stopPingInterval(); // S'assurer qu'il n'y a pas déjà un interval
      this.pingInterval = setInterval(() => {
        this.ping();
      }, 30000); // Ping toutes les 30 secondes
    }
  
    private stopPingInterval() {
      if (this.pingInterval) {
        clearInterval(this.pingInterval);
        this.pingInterval = null;
      }
    }
  
    private send(data: any) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(data));
      } else {
        console.warn('WebSocket non connecté, impossible d\'envoyer:', data);
      }
    }
  
    disconnect() {
      this.stopPingInterval();
      if (this.ws) {
        this.ws.close(1000, 'Disconnection requested'); // Code 1000 = fermeture normale
        this.ws = null;
      }
    }
  
    isConnected(): boolean {
      return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }
  
    getReadyState(): number | null {
      return this.ws ? this.ws.readyState : null;
    }
  
    forceReconnect() {
      this.reconnectAttempts = 0;
      this.disconnect();
      if (this.token) {
        setTimeout(() => {
          this.connect(this.token!, this.callbacks);
        }, 1000);
      }
    }
  }
  
  // Instance singleton
  let wsManager: WebSocketManager | null = null;
  
  export const getWebSocketManager = () => {
    if (!wsManager) {
      wsManager = new WebSocketManager();
    }
    return wsManager;
  };
  
  export default WebSocketManager;