'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { workflowService } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';

// Modal simple
function ConfirmModal({
  isOpen,
  title,
  message,
  onConfirm,
  onCancel,
}: {
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-lg max-w-sm w-full p-6 space-y-4">
        <h2 className="text-lg font-semibold">{title}</h2>
        <p>{message}</p>
        <div className="flex justify-end space-x-2">
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 transition"
          >
            Annuler
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition flex items-center"
          >
            {/** Vous pouvez ajouter un spinner ici si vous voulez */}
            Supprimer
          </button>
        </div>
      </div>
    </div>
  );
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  workflow_type: string;
  status: string;
  created_at: string;
}

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [toDeleteId, setToDeleteId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    if (!isAuthenticated) return;
    workflowService.getWorkflows()
      .then(data => setWorkflows(data))
      .catch(err => setError(err.error || 'Erreur chargement'))
      .finally(() => setLoading(false));
  }, [isAuthenticated]);

  const openDeleteModal = (id: string) => {
    setToDeleteId(id);
    setModalOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (!toDeleteId) return;
    setDeletingId(toDeleteId);
    setModalOpen(false);
    try {
      await workflowService.deleteWorkflow(toDeleteId);
      setWorkflows(ws => ws.filter(w => w.id !== toDeleteId));
    } catch (err: any) {
      setError(err.error || 'Erreur suppression');
    } finally {
      setDeletingId(null);
      setToDeleteId(null);
    }
  };

  return (
    <div className="container mx-auto p-4">
      {/* … ta barre de nav, header, etc. … */}

      {error && (
        <div className="bg-red-100 text-red-800 p-3 rounded mb-4">
          {error}
        </div>
      )}

      {loading ? (
        <div>Chargement…</div>
      ) : (
        <div className="space-y-4">
          {workflows.map(wf => (
            <div key={wf.id} className="bg-white p-4 rounded-lg shadow flex justify-between items-center">
              <div>
                <h3 className="text-lg font-medium">{wf.name}</h3>
                <p className="text-sm text-gray-500">{wf.description}</p>
              </div>
              <div className="flex space-x-2">
                <Link
                  href={`/workflows/${wf.id}`}
                  className="px-3 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 transition"
                >
                  Détails
                </Link>
                <button
                  onClick={() => openDeleteModal(wf.id)}
                  disabled={deletingId === wf.id}
                  className={`px-3 py-1 rounded transition ${
                    deletingId === wf.id
                      ? 'bg-red-200 text-red-400 cursor-not-allowed'
                      : 'bg-red-100 text-red-800 hover:bg-red-200'
                  }`}
                >
                  {deletingId === wf.id ? 'Suppression…' : 'Supprimer'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <ConfirmModal
        isOpen={modalOpen}
        title="Confirmer la suppression"
        message="Cette action est irréversible. Voulez-vous vraiment supprimer ce workflow ?"
        onConfirm={handleConfirmDelete}
        onCancel={() => setModalOpen(false)}
      />
    </div>
  );
}
