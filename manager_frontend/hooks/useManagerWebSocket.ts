'use client';

import { useEffect, useRef } from 'react';
import {
  ManagerRealtimeClient,
  RealtimeEvent,
  RealtimeHandlers,
} from '@/lib/realtime';

export interface UseManagerWebSocketOptions {
  workflowId?: string;
  /** Toasts gérés par le module realtime (défaut: true). */
  showToasts?: boolean;
  handlers?: RealtimeHandlers;
  /** Callback bas niveau (optionnel) — préférer handlers. */
  onEvent?: (event: RealtimeEvent) => void;
  enabled?: boolean;
}

/**
 * Connexion WebSocket Manager : reconnexion auto, ping, toasts cohérents.
 */
export function useManagerWebSocket(options: UseManagerWebSocketOptions = {}) {
  const {
    workflowId,
    showToasts = true,
    handlers,
    onEvent,
    enabled = true,
  } = options;

  const handlersRef = useRef(handlers);
  const onEventRef = useRef(onEvent);
  handlersRef.current = handlers;
  onEventRef.current = onEvent;

  useEffect(() => {
    if (!enabled || typeof window === 'undefined') return;

    const token = localStorage.getItem('token');
    if (!token) return;

    const client = new ManagerRealtimeClient();

    const wrapped: RealtimeHandlers = {
      onConnected: () => handlersRef.current?.onConnected?.(),
      onDisconnected: () => handlersRef.current?.onDisconnected?.(),
      onWorkflowStatus: (event) => {
        onEventRef.current?.(event);
        handlersRef.current?.onWorkflowStatus?.(event);
      },
      onWorkflowUpdate: (event) => {
        onEventRef.current?.(event);
        handlersRef.current?.onWorkflowUpdate?.(event);
      },
      onTaskStatus: (event) => {
        onEventRef.current?.(event);
        handlersRef.current?.onTaskStatus?.(event);
      },
      onTaskProgress: (event) => {
        onEventRef.current?.(event);
        handlersRef.current?.onTaskProgress?.(event);
      },
      onVolunteerStatus: (event) => {
        onEventRef.current?.(event);
        handlersRef.current?.onVolunteerStatus?.(event);
      },
      onVolunteerUpdate: (event) => {
        onEventRef.current?.(event);
        handlersRef.current?.onVolunteerUpdate?.(event);
      },
    };

    client.connect({
      token,
      workflowId,
      handlers: wrapped,
      showToasts,
    });

    return () => client.disconnect();
  }, [workflowId, showToasts, enabled]);
}
