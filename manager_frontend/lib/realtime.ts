/**
 * Module temps réel Manager — WebSocket propre, toasts cohérents, reconnexion auto.
 */

import { toast, TypeOptions } from 'react-toastify';

export type RealtimeEventType =
  | 'connection_established'
  | 'subscription_confirmed'
  | 'pong'
  | 'workflow_status_change'
  | 'workflow_update'
  | 'task_status_change'
  | 'task_status_update'
  | 'task_update'
  | 'task_status'
  | 'task_progress'
  | 'volunteer_update'
  | 'volunteer_status'
  | 'error';

export interface RealtimeEvent {
  type?: string;
  workflow_id?: string;
  task_id?: string;
  status?: string;
  message?: string;
  progress?: number;
  workflow?: { id?: string; status?: string; [key: string]: unknown };
  task?: { id?: string; status?: string; progress?: number; [key: string]: unknown };
  action?: string;
  timestamp?: number;
  [key: string]: unknown;
}

export interface RealtimeHandlers {
  onWorkflowStatus?: (event: RealtimeEvent) => void;
  onWorkflowUpdate?: (event: RealtimeEvent) => void;
  onTaskStatus?: (event: RealtimeEvent) => void;
  onTaskProgress?: (event: RealtimeEvent) => void;
  onVolunteerStatus?: (event: RealtimeEvent) => void;
  onVolunteerUpdate?: (event: RealtimeEvent) => void;
  onConnected?: () => void;
  onDisconnected?: () => void;
}

const STATUS_LABELS: Record<string, string> = {
  CREATED: 'Créé',
  VALIDATED: 'Validé',
  SUBMITTED: 'Soumis',
  SPLITTING: 'Découpage…',
  SPLIT_COMPLETED: 'Découpage terminé',
  ASSIGNING: 'Attribution…',
  PENDING: 'En attente de volontaires',
  WAITING_VOLUNTEERS: 'En attente de volontaires',
  RUNNING: 'En cours',
  PAUSED: 'En pause',
  AGGREGATING: 'Agrégation…',
  COMPLETED: 'Terminé',
  FAILED: 'Échoué',
  PARTIAL_FAILURE: 'Échec partiel',
  ERROR: 'Erreur',
  ASSIGNED: 'Assignée',
  STARTED: 'Démarrée',
};

const SUCCESS_STATUSES = new Set(['COMPLETED', 'SPLIT_COMPLETED']);
const ERROR_STATUSES = new Set(['FAILED', 'ERROR', 'PARTIAL_FAILURE']);
const INFO_STATUSES = new Set([
  'SUBMITTED',
  'SPLITTING',
  'ASSIGNING',
  'PENDING',
  'WAITING_VOLUNTEERS',
  'RUNNING',
  'AGGREGATING',
  'ASSIGNED',
  'STARTED',
  'CREATED',
]);

function statusLabel(status?: string): string {
  if (!status) return '';
  return STATUS_LABELS[status] || status;
}

function toastTone(status?: string): TypeOptions {
  if (!status) return 'default';
  if (SUCCESS_STATUSES.has(status)) return 'success';
  if (ERROR_STATUSES.has(status)) return 'error';
  if (INFO_STATUSES.has(status)) return 'info';
  return 'default';
}

function showToast(message: string, status?: string) {
  const tone = toastTone(status);
  const options = { position: 'top-right' as const, autoClose: 4500 };
  if (tone === 'success') toast.success(message, options);
  else if (tone === 'error') toast.error(message, options);
  else if (tone === 'info') toast.info(message, options);
  else toast(message, options);
}

/** Messages techniques à ignorer (pas de toast). */
const SILENT_TYPES = new Set([
  'connection_established',
  'subscription_confirmed',
  'pong',
  'ping',
]);

export function describeWorkflowEvent(event: RealtimeEvent): string | null {
  const status = event.status || event.workflow?.status;
  const label = statusLabel(status);
  if (event.message && typeof event.message === 'string' && event.message.trim()) {
    return event.message.trim();
  }
  if (label) return `Workflow : ${label}`;
  return null;
}

export function describeTaskEvent(event: RealtimeEvent): string | null {
  const status = event.status || event.task?.status;
  const label = statusLabel(status);
  if (event.message && typeof event.message === 'string' && event.message.trim()) {
    return event.message.trim();
  }
  if (label) return `Tâche : ${label}`;
  return null;
}

export function handleRealtimeEvent(
  event: RealtimeEvent,
  handlers: RealtimeHandlers = {},
  options: { workflowId?: string; showToasts?: boolean } = {},
) {
  if (!event || typeof event !== 'object') return;

  const type = String(event.type || '');
  if (!type || SILENT_TYPES.has(type)) {
    if (type === 'connection_established') handlers.onConnected?.();
    return;
  }

  const { workflowId, showToasts = true } = options;
  const scoped =
    !workflowId ||
    !event.workflow_id ||
    String(event.workflow_id) === String(workflowId) ||
    String(event.workflow?.id || '') === String(workflowId);

  if (type === 'workflow_status_change') {
    if (scoped) {
      handlers.onWorkflowStatus?.(event);
      if (showToasts) {
        const msg = describeWorkflowEvent(event);
        if (msg) showToast(msg, event.status);
      }
    }
    return;
  }

  if (type === 'workflow_update') {
    if (scoped) {
      handlers.onWorkflowUpdate?.(event);
      if (showToasts && event.action && event.action !== 'updated') {
        const status = event.workflow?.status || event.status;
        const msg = event.message || describeWorkflowEvent(event);
        if (msg) showToast(String(msg), status);
      }
    }
    return;
  }

  if (
    type === 'task_status_change' ||
    type === 'task_status_update' ||
    type === 'task_update' ||
    type === 'task_status'
  ) {
    if (scoped) {
      handlers.onTaskStatus?.(event);
      if (showToasts) {
        const status = event.status || event.task?.status;
        // Ne pas spammer pour chaque micro-statut
        if (status && (SUCCESS_STATUSES.has(status) || ERROR_STATUSES.has(status) || status === 'RUNNING' || status === 'STARTED')) {
          const msg = describeTaskEvent(event);
          if (msg) showToast(msg, status);
        }
      }
    }
    return;
  }

  if (type === 'task_progress') {
    if (scoped) handlers.onTaskProgress?.(event);
    return;
  }

  if (type === 'volunteer_status') {
    handlers.onVolunteerStatus?.(event);
    return;
  }

  if (type === 'volunteer_update') {
    handlers.onVolunteerUpdate?.(event);
    return;
  }

  // Autres événements : pas de toast d'erreur
}

function buildWsUrl(token: string): string {
  const base =
    process.env.NEXT_PUBLIC_MANAGER_WS_URL ||
    (typeof window !== 'undefined'
      ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/manager/`
      : 'ws://localhost:8002/ws/manager/');
  const sep = base.includes('?') ? '&' : '?';
  return `${base}${sep}token=${encodeURIComponent(token)}`;
}

export class ManagerRealtimeClient {
  private ws: WebSocket | null = null;
  private token: string | null = null;
  private handlers: RealtimeHandlers = {};
  private workflowId?: string;
  private showToasts = true;
  private reconnectAttempts = 0;
  private maxReconnect = 8;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private intentionalClose = false;

  connect(options: {
    token: string;
    workflowId?: string;
    handlers?: RealtimeHandlers;
    showToasts?: boolean;
  }) {
    this.token = options.token;
    this.workflowId = options.workflowId;
    this.handlers = options.handlers || {};
    this.showToasts = options.showToasts !== false;
    this.intentionalClose = false;
    this.openSocket();
  }

  private openSocket() {
    if (!this.token || typeof window === 'undefined') return;
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const url = buildWsUrl(this.token);
    const socket = new WebSocket(url);
    this.ws = socket;

    socket.onopen = () => {
      this.reconnectAttempts = 0;
      this.startPing();
      if (this.workflowId) {
        this.send({ type: 'subscribe_workflow', workflow_id: this.workflowId });
      }
      this.handlers.onConnected?.();
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as RealtimeEvent;
        handleRealtimeEvent(data, this.handlers, {
          workflowId: this.workflowId,
          showToasts: this.showToasts,
        });
      } catch {
        // JSON invalide : ignorer sans toast
      }
    };

    socket.onerror = () => {
      // onclose gère la reconnexion
    };

    socket.onclose = () => {
      this.stopPing();
      this.handlers.onDisconnected?.();
      if (!this.intentionalClose) this.scheduleReconnect();
    };
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnect) return;
    this.reconnectAttempts += 1;
    const delay = Math.min(1000 * 2 ** (this.reconnectAttempts - 1), 15000);
    this.reconnectTimer = setTimeout(() => this.openSocket(), delay);
  }

  private startPing() {
    this.stopPing();
    this.pingTimer = setInterval(() => {
      this.send({ type: 'ping' });
    }, 25000);
  }

  private stopPing() {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  send(payload: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  subscribeWorkflow(workflowId: string) {
    this.workflowId = workflowId;
    this.send({ type: 'subscribe_workflow', workflow_id: workflowId });
  }

  disconnect() {
    this.intentionalClose = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.stopPing();
    this.ws?.close(1000, 'client disconnect');
    this.ws = null;
  }

  get isConnected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
