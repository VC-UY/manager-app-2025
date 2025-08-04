import { useEffect } from 'react';

export function useManagerWebSocket(onEvent: (event: any) => void) {
  useEffect(() => {
    // Adapter l'URL à celle de ton backend WebSocket
    // Récupérer le token d'authentification depuis le localStorage
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    let wsUrl = process.env.NEXT_PUBLIC_MANAGER_WS_URL || 'ws://localhost:8001/ws/manager/';
    if (token) {
      wsUrl += (wsUrl.includes('?') ? '&' : '?') + `token=${encodeURIComponent(token)}`;
    }
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('[WebSocket] Connecté au manager backend');
    };
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('[WebSocket] Message reçu:', data);
        onEvent(data);
      } catch (e) {
        console.error('[WebSocket] Erreur de parsing:', e);
      }
    };
    socket.onerror = (err) => {
      console.error('[WebSocket] Erreur:', err);
    };
    socket.onclose = () => {
      console.warn('[WebSocket] Déconnecté du manager backend');
    };
    return () => {
      socket.close();
    };
  }, [onEvent]);
}
