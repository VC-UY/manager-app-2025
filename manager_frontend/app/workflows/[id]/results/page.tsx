'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { workflowService } from '@/lib/api';

interface OutputFile {
  name: string;
  path: string;
  size: number;
  modified: number;
}

interface OutputsResponse {
  files: OutputFile[];
  output_path?: string;
  workflow_id?: string;
  workflow_name?: string;
  status?: string;
  message?: string;
}

function formatBytes(bytes: number): string {
  if (!bytes || bytes < 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let n = bytes;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatDate(ts?: number): string {
  if (!ts) return '—';
  return new Intl.DateTimeFormat('fr-FR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(ts * 1000));
}

export default function WorkflowResultsPage() {
  const { id } = useParams();
  const [data, setData] = useState<OutputsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    try {
      const res = await workflowService.getWorkflowOutputs(id as string);
      setData(res);
      setError(null);
    } catch (err: any) {
      setError(err?.error || err?.message || 'Impossible de charger les résultats');
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  const handleDownloadFile = async (filePath: string) => {
    if (!id) return;
    setDownloading(filePath);
    try {
      const blob = await workflowService.downloadOutputFile(id as string, filePath);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filePath.split('/').pop() || 'output';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      toast.success('Téléchargement démarré', { position: 'top-right' });
    } catch (err: any) {
      toast.error(err?.error || 'Échec du téléchargement', { position: 'top-right' });
    } finally {
      setDownloading(null);
    }
  };

  const handleDownloadZip = async () => {
    if (!id) return;
    setDownloading('zip');
    try {
      const blob = await workflowService.downloadOutputsZip(id as string);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${data?.workflow_name || 'workflow'}_outputs.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      toast.success('Archive téléchargée', { position: 'top-right' });
    } catch (err: any) {
      toast.error(err?.error || 'Échec du téléchargement ZIP', { position: 'top-right' });
    } finally {
      setDownloading(null);
    }
  };

  const files = data?.files || [];

  return (
    <div
      className="min-h-screen"
      style={{ background: 'linear-gradient(135deg, #0A1628 0%, #1A2942 50%, #0A1628 100%)' }}
    >
      <ToastContainer theme="dark" />
      <div className="max-w-5xl mx-auto px-4 py-8">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
          <div>
            <Link
              href={`/workflows/${id}`}
              className="inline-flex items-center text-sm mb-3"
              style={{ color: '#60A5FA' }}
            >
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Retour au workflow
            </Link>
            <h1 className="text-2xl md:text-3xl font-bold" style={{ color: '#FFFFFF' }}>
              Résultats
            </h1>
            <p className="mt-1 text-sm" style={{ color: '#94A3B8' }}>
              {data?.workflow_name || 'Workflow'}
              {data?.status ? ` · ${data.status}` : ''}
            </p>
          </div>
          {files.length > 0 && (
            <button
              type="button"
              onClick={handleDownloadZip}
              disabled={downloading === 'zip'}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium"
              style={{
                background: 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
                color: '#FFFFFF',
                opacity: downloading === 'zip' ? 0.7 : 1,
              }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              {downloading === 'zip' ? 'Préparation…' : 'Tout télécharger (ZIP)'}
            </button>
          )}
        </div>

        {loading && (
          <div className="rounded-xl p-8 text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', color: '#94A3B8' }}>
            Chargement des résultats…
          </div>
        )}

        {!loading && error && (
          <div
            className="rounded-xl p-6 border-l-4"
            style={{
              background: 'rgba(239, 68, 68, 0.1)',
              borderColor: '#EF4444',
              color: '#FCA5A5',
            }}
          >
            {error}
          </div>
        )}

        {!loading && !error && files.length === 0 && (
          <div
            className="rounded-xl p-8 text-center"
            style={{
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(51, 65, 85, 0.8)',
              color: '#94A3B8',
            }}
          >
            <p className="text-lg mb-2" style={{ color: '#E2E8F0' }}>Aucun fichier de sortie</p>
            <p className="text-sm">
              {data?.message ||
                'Les résultats apparaîtront ici une fois le workflow terminé et les sorties agrégées.'}
            </p>
            <button
              type="button"
              onClick={load}
              className="mt-4 px-4 py-2 rounded-lg text-sm"
              style={{ background: 'rgba(59, 130, 246, 0.2)', color: '#60A5FA' }}
            >
              Actualiser
            </button>
          </div>
        )}

        {!loading && files.length > 0 && (
          <div
            className="rounded-xl overflow-hidden"
            style={{
              background: 'rgba(30, 41, 59, 0.7)',
              border: '1px solid rgba(51, 65, 85, 0.9)',
            }}
          >
            <div className="px-5 py-4 border-b" style={{ borderColor: 'rgba(51, 65, 85, 0.9)' }}>
              <span className="text-sm font-medium" style={{ color: '#E2E8F0' }}>
                {files.length} fichier{files.length > 1 ? 's' : ''}
              </span>
            </div>
            <ul className="divide-y" style={{ borderColor: 'rgba(51, 65, 85, 0.6)' }}>
              {files.map((file) => (
                <li
                  key={file.path}
                  className="flex flex-wrap items-center justify-between gap-3 px-5 py-4"
                >
                  <div className="min-w-0">
                    <p className="font-medium truncate" style={{ color: '#FFFFFF' }}>
                      {file.path}
                    </p>
                    <p className="text-xs mt-1" style={{ color: '#94A3B8' }}>
                      {formatBytes(file.size)} · modifié {formatDate(file.modified)}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleDownloadFile(file.path)}
                    disabled={downloading === file.path}
                    className="shrink-0 inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm"
                    style={{
                      background: 'rgba(59, 130, 246, 0.15)',
                      border: '1px solid rgba(59, 130, 246, 0.35)',
                      color: '#60A5FA',
                      opacity: downloading === file.path ? 0.6 : 1,
                    }}
                  >
                    Télécharger
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
