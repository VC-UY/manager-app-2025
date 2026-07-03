'use client';

import Image from 'next/image';
import Link from 'next/link';

const steps = [
  {
    title: 'Creer un workflow',
    text: 'Definissez votre calcul, importez les donnees et lancez la soumission.',
  },
  {
    title: 'Distribution automatique',
    text: 'Le coordinateur repartit les taches sur les machines volontaires disponibles.',
  },
  {
    title: 'Suivre les resultats',
    text: 'Consultez l avancement et telechargez les sorties une fois le traitement termine.',
  },
];

export default function Home() {
  return (
    <main
      className="min-h-screen text-white"
      style={{ background: 'linear-gradient(180deg, #001440 0%, #002060 50%, #001440 100%)' }}
    >
      <nav
        className="sticky top-0 z-50 border-b backdrop-blur-xl"
        style={{
          background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.9) 0%, rgba(0, 20, 64, 0.9) 100%)',
          borderColor: 'rgba(0, 180, 240, 0.25)',
        }}
      >
        <div className="container mx-auto flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <Image src="/logo.svg" alt="VolunSys-UY1" width={44} height={44} className="rounded-xl" priority />
            <div>
              <p className="text-lg font-bold leading-tight">VolunSys-UY1</p>
              <p className="text-xs text-cyan-300">Application Manager</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Link href="/login" className="text-sm text-white/80 hover:text-white">
              Connexion
            </Link>
            <Link
              href="/register"
              className="rounded-xl px-5 py-2 text-sm font-bold text-white"
              style={{
                background: 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                boxShadow: '0 8px 24px rgba(0, 180, 240, 0.35)',
              }}
            >
              S inscrire
            </Link>
          </div>
        </div>
      </nav>

      <section className="container mx-auto grid items-center gap-12 px-6 py-20 lg:grid-cols-2">
        <div>
          <p className="mb-4 text-sm font-semibold uppercase tracking-widest text-cyan-400">
            Calcul volontaire
          </p>
          <h1 className="text-4xl font-extrabold leading-tight md:text-5xl">
            <span
              style={{
                background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Soumettez vos calculs
            </span>
            <br />
            <span className="text-cyan-300">sur le reseau VolunSys</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg text-white/85">
            Interface manager pour creer des workflows, suivre les taches et recuperer les resultats
            sur l infrastructure de calcul volontaire.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <Link
              href="/register"
              className="rounded-xl px-8 py-4 text-sm font-bold text-white"
              style={{
                background: 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                boxShadow: '0 8px 32px rgba(0, 180, 240, 0.35)',
              }}
            >
              Commencer
            </Link>
            <Link
              href="/login"
              className="rounded-xl border px-8 py-4 text-sm font-bold text-white"
              style={{ borderColor: 'rgba(0,212,255,0.5)', background: 'rgba(255,255,255,0.06)' }}
            >
              Se connecter
            </Link>
          </div>
        </div>

        <div className="rounded-2xl border p-2" style={{ borderColor: 'rgba(0,180,240,0.3)' }}>
          <Image
            src="/images/architecture.svg"
            alt="Schema Manager, Coordinateur et Volontaires"
            width={640}
            height={400}
            className="w-full rounded-xl"
            priority
          />
        </div>
      </section>

      <section className="container mx-auto px-6 pb-20">
        <h2 className="mb-8 text-center text-2xl font-bold text-white">Comment ca marche</h2>
        <div className="grid gap-6 md:grid-cols-3">
          {steps.map((step, index) => (
            <div
              key={step.title}
              className="rounded-2xl p-6"
              style={{
                background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.7) 0%, rgba(0, 20, 64, 0.7) 100%)',
                border: '2px solid rgba(0, 180, 240, 0.3)',
              }}
            >
              <p className="mb-3 text-sm font-bold text-cyan-300">Etape {index + 1}</p>
              <h3 className="text-lg font-bold text-white">{step.title}</h3>
              <p className="mt-2 text-sm text-white/75">{step.text}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t py-8 text-center text-sm text-cyan-300/80" style={{ borderColor: 'rgba(0,180,240,0.2)' }}>
        VolunSys-UY1 Manager, Universite de Yaounde I
      </footer>
    </main>
  );
}
