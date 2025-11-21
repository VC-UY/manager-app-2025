// app/login/page.tsx
'use client';

import { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import Link from 'next/link';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [formError, setFormError] = useState('');
  const { login, loading, error } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError('');

    if (!email || !password) {
      setFormError('Veuillez remplir tous les champs');
      return;
    }

    try {
      await login(email, password);
    } catch (err: any) {
      console.error('Erreur de connexion:', err);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden flex flex-col justify-center py-12 sm:px-6 lg:px-8"
      style={{
        background: 'linear-gradient(135deg, #001440 0%, #002060 50%, #001440 100%)',
      }}>
      
      {/* Animated Background Elements */}
      <div className="absolute top-0 right-0 w-96 h-96 rounded-full opacity-20"
        style={{
          background: 'radial-gradient(circle, rgba(0, 212, 255, 0.15) 0%, transparent 70%)',
          animation: 'pulse 8s ease-in-out infinite',
        }} />
      <div className="absolute bottom-0 left-0 w-80 h-80 rounded-full opacity-20"
        style={{
          background: 'radial-gradient(circle, rgba(0, 180, 240, 0.1) 0%, transparent 70%)',
          animation: 'pulse 6s ease-in-out infinite 1s',
        }} />

      <style jsx>{`
        @keyframes pulse {
          0%, 100% { transform: scale(1); opacity: 0.5; }
          50% { transform: scale(1.1); opacity: 0.8; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
          50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.6); }
        }
      `}</style>

      <div className="sm:mx-auto sm:w-full sm:max-w-md relative z-10">
        {/* Logo */}
        <div className="flex justify-center mb-6">
          <div className="relative"
            style={{
              animation: 'float 3s ease-in-out infinite',
            }}>
            <div className="h-20 w-20 rounded-2xl flex items-center justify-center text-white font-bold text-2xl relative overflow-hidden"
              style={{
                background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.25) 0%, rgba(0, 180, 240, 0.15) 100%)',
                border: '3px solid rgba(0, 212, 255, 0.4)',
                boxShadow: '0 8px 32px rgba(0, 212, 255, 0.3)',
              }}>
              <span style={{
                background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontSize: '28px',
                fontWeight: 800,
                letterSpacing: '1px',
              }}>CV</span>
              <div className="absolute inset-0"
                style={{
                  background: 'radial-gradient(circle at 0% 50%, rgba(0, 212, 255, 0.2) 0%, transparent 60%)',
                  animation: 'pulse 2s ease-in-out infinite',
                }} />
            </div>
          </div>
        </div>

        {/* Title */}
        <h2 className="text-center text-4xl font-extrabold mb-3"
          style={{
            background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            letterSpacing: '0.5px',
          }}>
          Connexion
        </h2>
        <p className="text-center text-sm font-medium"
          style={{
            color: '#00B0F0',
            letterSpacing: '0.3px',
          }}>
          Ou{' '}
          <Link href="/register" 
            className="font-semibold transition-all duration-300"
            style={{
              color: '#00D4FF',
            }}
            onMouseEnter={(e) => e.currentTarget.style.textShadow = '0 0 10px rgba(0, 212, 255, 0.8)'}
            onMouseLeave={(e) => e.currentTarget.style.textShadow = 'none'}>
            créez un nouveau compte
          </Link>
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md relative z-10">
        <div className="py-10 px-8 shadow-2xl rounded-3xl backdrop-blur-xl"
          style={{
            background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.9) 0%, rgba(0, 20, 64, 0.9) 100%)',
            border: '2px solid rgba(0, 180, 240, 0.3)',
            boxShadow: '0 12px 48px rgba(0, 32, 96, 0.5)',
          }}>
          
          {/* Error Alert */}
          {(formError || error) && (
            <div className="mb-6 p-4 rounded-xl border-l-4"
              style={{
                background: 'linear-gradient(90deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 68, 68, 0.05) 100%)',
                borderColor: '#FF4444',
              }}>
              <p className="text-sm font-medium" style={{ color: '#FFB4B4' }}>
                {formError || error}
              </p>
            </div>
          )}
          
          <form className="space-y-6" onSubmit={handleSubmit}>
            {/* Email Input */}
            <div>
              <label htmlFor="email" className="block text-sm font-semibold mb-2"
                style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>
                Adresse email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="appearance-none block w-full px-4 py-3 rounded-xl shadow-sm text-white transition-all duration-300"
                placeholder="votre@email.com"
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '2px solid rgba(0, 180, 240, 0.3)',
                  backdropFilter: 'blur(10px)',
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = '#00D4FF';
                  e.target.style.boxShadow = '0 4px 20px rgba(0, 212, 255, 0.3)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                  e.target.style.boxShadow = 'none';
                }}
              />
            </div>

            {/* Password Input */}
            <div>
              <label htmlFor="password" className="block text-sm font-semibold mb-2"
                style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>
                Mot de passe
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="appearance-none block w-full px-4 py-3 rounded-xl shadow-sm text-white transition-all duration-300"
                placeholder="••••••••"
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '2px solid rgba(0, 180, 240, 0.3)',
                  backdropFilter: 'blur(10px)',
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = '#00D4FF';
                  e.target.style.boxShadow = '0 4px 20px rgba(0, 212, 255, 0.3)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                  e.target.style.boxShadow = 'none';
                }}
              />
            </div>

            {/* Remember & Forgot */}
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  id="remember_me"
                  name="remember_me"
                  type="checkbox"
                  className="h-4 w-4 rounded transition-all"
                  style={{
                    accentColor: '#00D4FF',
                  }}
                />
                <label htmlFor="remember_me" className="ml-2 block text-sm font-medium"
                  style={{ color: '#FFFFFF' }}>
                  Se souvenir de moi
                </label>
              </div>

              <div className="text-sm">
                <a href="#" className="font-semibold transition-all duration-300"
                  style={{ color: '#00B0F0' }}
                  onMouseEnter={(e) => e.currentTarget.style.color = '#00D4FF'}
                  onMouseLeave={(e) => e.currentTarget.style.color = '#00B0F0'}>
                  Mot de passe oublié?
                </a>
              </div>
            </div>

            {/* Submit Button */}
            <div>
              <button
                type="submit"
                disabled={loading}
                className="w-full flex justify-center items-center py-3 px-4 rounded-xl text-sm font-bold text-white transition-all duration-300 relative overflow-hidden"
                style={{
                  background: loading 
                    ? 'linear-gradient(135deg, rgba(0, 180, 240, 0.5) 0%, rgba(0, 212, 255, 0.5) 100%)'
                    : 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                  border: '2px solid rgba(0, 212, 255, 0.4)',
                  boxShadow: '0 8px 24px rgba(0, 180, 240, 0.3)',
                  letterSpacing: '0.5px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                }}
                onMouseEnter={(e) => {
                  if (!loading) {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 12px 32px rgba(0, 212, 255, 0.5)';
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 180, 240, 0.3)';
                }}>
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Connexion en cours...
                  </>
                ) : (
                  'Se connecter'
                )}
              </button>
            </div>
          </form>

          {/* Divider */}
          <div className="mt-8">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t"
                  style={{
                    borderColor: 'rgba(0, 180, 240, 0.2)',
                  }} />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-3 text-sm font-medium"
                  style={{
                    background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.9) 0%, rgba(0, 20, 64, 0.9) 100%)',
                    color: '#00B0F0',
                  }}>
                  Ou revenir à
                </span>
              </div>
            </div>

            {/* Home Button */}
            <div className="mt-6">
              <Link
                href="/"
                className="w-full flex justify-center py-3 px-4 rounded-xl text-sm font-semibold transition-all duration-300"
                style={{
                  background: 'rgba(255, 255, 255, 0.08)',
                  border: '2px solid rgba(0, 180, 240, 0.3)',
                  color: '#FFFFFF',
                  backdropFilter: 'blur(10px)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(0, 180, 240, 0.15)';
                  e.currentTarget.style.borderColor = '#00D4FF';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                  e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}>
                Page d'accueil
              </Link>
            </div>
          </div>
        </div>

        {/* Footer Text */}
        <div className="mt-6 text-center">
          <p className="text-xs font-medium"
            style={{
              color: '#00B0F0',
              letterSpacing: '0.5px',
            }}>
            La puissance collective au service du calcul scientifique
          </p>
        </div>
      </div>
    </div>
  );
}