'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import { taskService } from '@/lib/api';
import { ManagerNav } from '@/components/ManagerNav';

type TaskRow = {
  id: string;
  name: string;
  status: string;
  progress?: number;
  workflow?: string;
};

export default function TasksPage() {
  const { isAuthenticated } = useAuth();
  const [tasks, setTasks] = useState<TaskRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState('all');

  useEffect(() => {
    if (!isAuthenticated) {
      setError('Veuillez vous connecter pour voir vos taches');
      setLoading(false);
      return;
    }
    const load = async () => {
      try {
        setLoading(true);
        const data =
          statusFilter === 'all'
            ? await taskService.getTasks()
            : await taskService.getTasksByStatus(statusFilter);
        setTasks(Array.isArray(data) ? data : []);
        setError(null);
      } catch (err: unknown) {
        const message = err && typeof err === 'object' && 'error' in err ? String((err as { error: string }).error) : 'Erreur de chargement';
        setError(message);
        setTasks([]);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [isAuthenticated, statusFilter]);

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #001440 0%, #002060 50%, #001440 100%)' }}>
      <div className="container mx-auto max-w-7xl px-4 py-8">
        <ManagerNav />
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <h1 className="text-3xl font-bold text-white">Mes taches</h1>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="rounded-xl border border-cyan-500/40 bg-black/30 px-4 py-2 text-sm text-white"
          >
            <option value="all">Tous les statuts</option>
            <option value="PENDING">En attente</option>
            <option value="ASSIGNED">Assignees</option>
            <option value="RUNNING">En cours</option>
            <option value="COMPLETED">Terminees</option>
            <option value="FAILED">Echouees</option>
          </select>
        </div>

        {loading && <p className="text-cyan-200">Chargement...</p>}
        {error && <p className="text-amber-300">{error}</p>}

        {!loading && tasks.length === 0 && !error && (
          <p className="rounded-2xl border border-cyan-500/30 bg-black/20 p-8 text-center text-white/70">
            Aucune tache pour le moment. Soumettez un workflow pour generer des taches.
          </p>
        )}

        <div className="space-y-3">
          {tasks.map((task) => (
            <div
              key={task.id}
              className="rounded-2xl border border-cyan-500/25 bg-black/20 p-5 backdrop-blur"
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold text-white">{task.name}</h2>
                  <p className="text-sm text-cyan-200/80">Statut : {task.status}</p>
                </div>
                {task.workflow && (
                  <Link
                    href={`/workflows/${task.workflow}`}
                    className="rounded-lg bg-cyan-500/20 px-4 py-2 text-sm text-cyan-200 hover:bg-cyan-500/30"
                  >
                    Voir le workflow
                  </Link>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
