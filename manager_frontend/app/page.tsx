'use client';

import Link from 'next/link';
import { useState } from 'react';

export default function Home() {
  const [activeFeature, setActiveFeature] = useState(0);
  
  const features = [
    {
      title: "Parallélisation Intelligente",
      description: "Notre système analyse automatiquement votre workflow et identifie les parties qui peuvent être exécutées en parallèle pour maximiser l'efficacité.",
    },
    {
      title: "Tolérance aux Pannes",
      description: "Le système détecte automatiquement les défaillances et réattribue les tâches à d'autres volontaires, garantissant la robustesse et la fiabilité de vos calculs.",
    },
    {
      title: "Surveillance en Temps Réel",
      description: "Visualisez l'avancement de vos tâches en temps réel avec des statistiques détaillées et des indicateurs de performance pour optimiser vos workflows.",
    }
  ];

  return (
    <main className="min-h-screen" style={{ background: 'linear-gradient(180deg, #001440 0%, #002060 50%, #001440 100%)' }}>
      <style jsx>{`
        @keyframes pulse {
          0%, 100% { transform: scale(1); opacity: 0.5; }
          50% { transform: scale(1.05); opacity: 0.8; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(5deg); }
        }
        @keyframes orbit {
          0% { transform: rotate(0deg) translateX(80px) rotate(0deg); }
          100% { transform: rotate(360deg) translateX(80px) rotate(-360deg); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
          50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.6); }
        }
        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      {/* Navbar */}
      <nav className="relative z-50 backdrop-blur-xl" style={{
        background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.8) 0%, rgba(0, 20, 64, 0.8) 100%)',
        borderBottom: '2px solid rgba(0, 180, 240, 0.2)',
        boxShadow: '0 4px 24px rgba(0, 32, 96, 0.5)',
      }}>
        <div className="container mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="relative h-12 w-12 rounded-xl flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.25) 0%, rgba(0, 180, 240, 0.15) 100%)',
                  border: '2px solid rgba(0, 212, 255, 0.4)',
                  animation: 'glow 3s ease-in-out infinite',
                }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="12" r="3" stroke="#00D4FF" strokeWidth="2"/>
                  <circle cx="12" cy="5" r="2" fill="#00D4FF"/>
                  <circle cx="12" cy="19" r="2" fill="#00D4FF"/>
                  <circle cx="5" cy="12" r="2" fill="#00B0F0"/>
                  <circle cx="19" cy="12" r="2" fill="#00B0F0"/>
                  <line x1="12" y1="9" x2="12" y2="7" stroke="#00D4FF" strokeWidth="1.5"/>
                  <line x1="12" y1="17" x2="12" y2="15" stroke="#00D4FF" strokeWidth="1.5"/>
                  <line x1="9" y1="12" x2="7" y2="12" stroke="#00B0F0" strokeWidth="1.5"/>
                  <line x1="17" y1="12" x2="15" y2="12" stroke="#00B0F0" strokeWidth="1.5"/>
                </svg>
              </div>
              <span className="text-xl font-bold" style={{
                background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                letterSpacing: '0.5px',
              }}>VolunSys-UY1</span>
            </div>
            
            <div className="hidden md:flex space-x-8">
              {['Comment ça marche', 'Applications', 'Fonctionnalités'].map((item, idx) => (
                <a key={idx} href={`#${item.toLowerCase().replace(/\s/g, '-').normalize('NFD').replace(/[\u0300-\u036f]/g, '')}`}
                  className="text-sm font-semibold transition-all duration-300"
                  style={{ color: '#00B0F0', letterSpacing: '0.3px' }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = '#00D4FF'; e.currentTarget.style.textShadow = '0 0 10px rgba(0, 212, 255, 0.6)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = '#00B0F0'; e.currentTarget.style.textShadow = 'none'; }}>
                  {item}
                </a>
              ))}
            </div>
            
            <div className="flex items-center gap-4">
              <Link href="/login" 
                className="text-sm font-semibold transition-all duration-300 px-4 py-2 rounded-lg"
                style={{ color: '#FFFFFF' }}
                onMouseEnter={(e) => { e.currentTarget.style.color = '#00D4FF'; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = '#FFFFFF'; }}>
                Connexion
              </Link>
              <Link href="/register" 
                className="px-6 py-2.5 rounded-xl text-sm font-bold transition-all duration-300"
                style={{
                  background: 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                  color: '#FFFFFF',
                  border: '2px solid rgba(0, 212, 255, 0.4)',
                  boxShadow: '0 4px 16px rgba(0, 180, 240, 0.3)',
                  letterSpacing: '0.5px',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 212, 255, 0.5)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 180, 240, 0.3)';
                }}>
                S'inscrire
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden pt-20 pb-32">
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

        {/* Computing Particles */}
        <div className="absolute inset-0 overflow-hidden opacity-30">
          {[...Array(15)].map((_, i) => (
            <div key={i} className="absolute text-xs font-mono"
              style={{
                color: '#00D4FF',
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animation: `float ${3 + Math.random() * 3}s ease-in-out infinite ${Math.random() * 2}s`,
              }}>
              {Math.random() > 0.5 ? '01' : '10'}
            </div>
          ))}
        </div>
        
        <div className="container mx-auto px-6 relative z-10">
          <div className="flex flex-col md:flex-row items-center justify-between gap-12">
            <div className="md:w-1/2 text-center md:text-left" style={{ animation: 'slideIn 0.8s ease-out' }}>
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-6"
                style={{
                  background: 'linear-gradient(90deg, rgba(0, 180, 240, 0.15) 0%, transparent 100%)',
                  border: '1px solid rgba(0, 212, 255, 0.3)',
                }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M13 10V3L4 14h7v7l9-11h-7z" fill="#00D4FF"/>
                </svg>
                <span style={{ color: '#00B0F0', fontSize: '14px', fontWeight: 600, letterSpacing: '0.5px' }}>
                  Plateforme de Calcul Volontaire
                </span>
              </div>

              <h1 className="text-5xl md:text-6xl font-extrabold mb-6 leading-tight">
                <span style={{
                  background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  letterSpacing: '1px',
                }}>
                  Calcul Distribué
                </span>
                <br />
                <span style={{ color: '#00B0F0', letterSpacing: '0.5px' }}>Simple et Puissant</span>
              </h1>
              
              <p className="text-xl mb-10 max-w-xl" style={{ color: '#FFFFFF', opacity: 0.9 }}>
                Une plateforme innovante qui orchestre intelligemment vos workflows de calcul intensif sur un réseau de 
                <span style={{ color: '#00D4FF', fontWeight: 700 }}> volontaires communautaires</span>.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center md:justify-start">
                <Link href="/register" 
                  className="flex items-center justify-center gap-2 px-8 py-4 rounded-xl font-bold transition-all duration-300"
                  style={{
                    background: 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                    color: '#FFFFFF',
                    border: '2px solid rgba(0, 212, 255, 0.4)',
                    boxShadow: '0 8px 32px rgba(0, 180, 240, 0.4)',
                    letterSpacing: '0.5px',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-3px)';
                    e.currentTarget.style.boxShadow = '0 12px 40px rgba(0, 212, 255, 0.6)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = '0 8px 32px rgba(0, 180, 240, 0.4)';
                  }}>
                  <span>Commencer gratuitement</span>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M5 12h14M12 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </Link>
                <Link href="#comment-ca-marche" 
                  className="flex items-center justify-center gap-2 px-8 py-4 rounded-xl font-bold transition-all duration-300"
                  style={{
                    background: 'rgba(255, 255, 255, 0.1)',
                    color: '#FFFFFF',
                    border: '2px solid rgba(0, 180, 240, 0.4)',
                    backdropFilter: 'blur(10px)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(0, 180, 240, 0.15)';
                    e.currentTarget.style.borderColor = '#00D4FF';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                    e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.4)';
                  }}>
                  <span>En savoir plus</span>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 5v14M5 12l7 7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </Link>
              </div>
            </div>
            
            {/* Animated Computing Visualization */}
            <div className="md:w-1/2 relative" style={{ animation: 'slideIn 0.8s ease-out 0.2s both' }}>
              <div className="relative h-96 w-full">
                {/* Central Hub */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-24 h-24 rounded-2xl flex items-center justify-center"
                  style={{
                    background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.25) 0%, rgba(0, 180, 240, 0.15) 100%)',
                    border: '3px solid rgba(0, 212, 255, 0.5)',
                    boxShadow: '0 0 40px rgba(0, 212, 255, 0.5)',
                    animation: 'pulse 3s ease-in-out infinite',
                  }}>
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="4" stroke="#00D4FF" strokeWidth="2"/>
                    <circle cx="12" cy="4" r="2" fill="#00D4FF"/>
                    <circle cx="12" cy="20" r="2" fill="#00D4FF"/>
                    <circle cx="4" cy="12" r="2" fill="#00B0F0"/>
                    <circle cx="20" cy="12" r="2" fill="#00B0F0"/>
                    <circle cx="7" cy="7" r="1.5" fill="#00B0F0"/>
                    <circle cx="17" cy="7" r="1.5" fill="#00B0F0"/>
                    <circle cx="7" cy="17" r="1.5" fill="#00B0F0"/>
                    <circle cx="17" cy="17" r="1.5" fill="#00B0F0"/>
                  </svg>
                </div>

                {/* Orbiting Volunteer Nodes */}
                {[0, 1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="absolute top-1/2 left-1/2"
                    style={{
                      animation: `orbit ${8 + i}s linear infinite ${i * 1.5}s`,
                    }}>
                    <div className="w-16 h-16 rounded-xl flex items-center justify-center"
                      style={{
                        background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.9) 0%, rgba(0, 20, 64, 0.9) 100%)',
                        border: '2px solid rgba(0, 180, 240, 0.4)',
                        boxShadow: '0 4px 16px rgba(0, 180, 240, 0.3)',
                      }}>
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="3" y="3" width="18" height="14" rx="2" stroke="#00B0F0" strokeWidth="2"/>
                        <path d="M8 21h8M12 17v4" stroke="#00B0F0" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                    </div>
                  </div>
                ))}

                {/* Data Streams */}
                {[...Array(12)].map((_, i) => (
                  <div key={i} className="absolute"
                    style={{
                      width: '2px',
                      height: '20px',
                      background: 'linear-gradient(180deg, #00D4FF 0%, transparent 100%)',
                      left: `${20 + Math.random() * 60}%`,
                      top: `${20 + Math.random() * 60}%`,
                      animation: `float ${2 + Math.random() * 2}s ease-in-out infinite ${Math.random() * 2}s`,
                      opacity: 0.6,
                    }} />
                ))}

                {/* Computing Stats Box */}
                <div className="absolute bottom-0 right-0 p-4 rounded-xl backdrop-blur-xl"
                  style={{
                    background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.9) 0%, rgba(0, 20, 64, 0.9) 100%)',
                    border: '2px solid rgba(0, 180, 240, 0.3)',
                    boxShadow: '0 8px 32px rgba(0, 32, 96, 0.5)',
                  }}>
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: '#00D4FF' }} />
                    <span style={{ color: '#00B0F0', fontSize: '12px', fontWeight: 600 }}>Système actif</span>
                  </div>
                  <div style={{ color: '#FFFFFF', fontSize: '20px', fontWeight: 700 }}>
                    Réseau de <span style={{ color: '#00D4FF' }}>volontaires</span>
                  </div>
                  <div style={{ color: '#00B0F0', fontSize: '12px', marginTop: '4px' }}>
                    Calcul distribué haute performance
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="comment-ca-marche" className="py-20 relative">
        <div className="absolute inset-0" style={{
          background: 'linear-gradient(180deg, rgba(0, 20, 64, 0.3) 0%, transparent 100%)',
        }} />
        
        <div className="container mx-auto px-6 relative z-10">
          <div className="text-center mb-16" style={{ animation: 'slideIn 0.8s ease-out' }}>
            <h2 className="text-4xl font-bold mb-4" style={{
              background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '0.5px',
            }}>
              Comment ça fonctionne
            </h2>
            <p className="text-xl max-w-2xl mx-auto" style={{ color: '#00B0F0' }}>
              Une approche en trois étapes pour transformer vos calculs complexes en tâches distribuées et efficaces.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                number: "1",
                title: "Créez votre workflow",
                desc: "Définissez votre workflow de calcul intensif, importez vos données et configurez les paramètres d'exécution selon vos besoins spécifiques.",
                features: ["Interface intuitive", "Import simple des données", "Estimation automatique"],
                color: "#00B0F0"
              },
              {
                number: "2",
                title: "Orchestration automatique",
                desc: "Notre système analyse vos tâches, les décompose intelligemment et les distribue aux nœuds de calcul volontaires optimaux disponibles.",
                features: ["Découpage intelligent", "Attribution optimale", "Gestion des dépendances"],
                color: "#00D4FF"
              },
              {
                number: "3",
                title: "Résultats consolidés",
                desc: "Suivez en temps réel l'avancement de chaque tâche et récupérez les résultats agrégés et validés une fois le traitement terminé.",
                features: ["Visualisation temps réel", "Agrégation intelligente", "Export multi-formats"],
                color: "#00B0F0"
              }
            ].map((step, idx) => (
              <div key={idx} className="relative p-8 rounded-2xl transition-all duration-300"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
                  border: '2px solid rgba(0, 180, 240, 0.3)',
                  backdropFilter: 'blur(10px)',
                  animation: `slideIn 0.8s ease-out ${idx * 0.2}s both`,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-8px)';
                  e.currentTarget.style.borderColor = step.color;
                  e.currentTarget.style.boxShadow = `0 12px 40px ${step.color}50`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                  e.currentTarget.style.boxShadow = 'none';
                }}>
                
                <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 w-12 h-12 rounded-full flex items-center justify-center font-bold text-xl"
                  style={{
                    background: `linear-gradient(135deg, ${step.color} 0%, rgba(0, 20, 64, 0.9) 100%)`,
                    border: `2px solid ${step.color}`,
                    boxShadow: `0 0 20px ${step.color}70`,
                    color: '#FFFFFF',
                  }}>
                  {step.number}
                </div>

                <div className="text-center mb-4 pt-8">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-4"
                    style={{
                      background: `linear-gradient(135deg, ${step.color}20 0%, ${step.color}10 100%)`,
                      border: `2px solid ${step.color}40`,
                    }}>
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      {idx === 0 && (
                        <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" stroke={step.color} strokeWidth="2"/>
                      )}
                      {idx === 1 && (
                        <>
                          <circle cx="12" cy="12" r="3" stroke={step.color} strokeWidth="2"/>
                          <path d="M12 1v6M12 17v6M23 12h-6M7 12H1" stroke={step.color} strokeWidth="2"/>
                        </>
                      )}
                      {idx === 2 && (
                        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" stroke={step.color} strokeWidth="2"/>
                      )}
                    </svg>
                  </div>
                </div>

                <h3 className="text-xl font-bold text-center mb-3" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>
                  {step.title}
                </h3>
                
                <p className="text-center mb-6" style={{ color: '#00B0F0', fontSize: '14px' }}>
                  {step.desc}
                </p>

                <div className="space-y-2">
                  {step.features.map((feature, fidx) => (
                    <div key={fidx} className="flex items-center gap-2 p-2 rounded-lg"
                      style={{ background: 'rgba(0, 180, 240, 0.1)' }}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M5 13l4 4L19 7" stroke={step.color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      <span style={{ color: '#FFFFFF', fontSize: '13px' }}>{feature}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Community Stats Section */}
      <section className="py-20 relative">
        <div className="container mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4" style={{
              background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              La Force de la Communauté
            </h2>
            <p style={{ color: '#00B0F0', fontSize: '16px' }}>
              Des milliers de volontaires unis pour la science
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { value: '⚡', label: 'Performances optimisées', color: '#00B0F0' },
              { value: '🚀', label: 'Calculs accélérés', color: '#00D4FF' },
              { value: '📈', label: 'Évolutivité flexible', color: '#00B0F0' }
            ].map((stat, idx) => (
              <div key={idx} className="p-8 rounded-2xl text-center transition-all duration-300"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
                  border: '2px solid rgba(0, 180, 240, 0.3)',
                  backdropFilter: 'blur(10px)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-5px) scale(1.05)';
                  e.currentTarget.style.borderColor = stat.color;
                  e.currentTarget.style.boxShadow = `0 12px 40px ${stat.color}50`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0) scale(1)';
                  e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                  e.currentTarget.style.boxShadow = 'none';
                }}>
                <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl mb-6"
                  style={{
                    background: `linear-gradient(135deg, ${stat.color}20 0%, ${stat.color}10 100%)`,
                    border: `2px solid ${stat.color}40`,
                  }}>
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    {idx === 0 && (
                      <path d="M13 10V3L4 14h7v7l9-11h-7z" fill={stat.color}/>
                    )}
                    {idx === 1 && (
                      <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" stroke={stat.color} strokeWidth="2"/>
                    )}
                    {idx === 2 && (
                      <path d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" stroke={stat.color} strokeWidth="2"/>
                    )}
                  </svg>
                </div>
                <div className="text-5xl font-bold mb-2" style={{ color: stat.color }}>
                  {stat.value}
                </div>
                <p style={{ color: '#FFFFFF', fontSize: '14px', fontWeight: 500 }}>
                  {stat.label}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Applications Section */}
      <section id="applications" className="py-20 relative">
        <div className="absolute inset-0" style={{
          background: 'linear-gradient(180deg, transparent 0%, rgba(0, 20, 64, 0.3) 100%)',
        }} />

        <div className="container mx-auto px-6 relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4" style={{
              background: 'linear-gradient(135deg, #FFFFFF 0%, #00D4FF 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              Applications
            </h2>
            <p className="text-xl max-w-2xl mx-auto" style={{ color: '#00B0F0' }}>
              Notre plateforme s'adapte à diverses applications de calcul intensif
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              {
                title: "Machine Learning",
                desc: "Distribuez l'entraînement de vos modèles sur plusieurs nœuds pour accélérer la convergence et traiter de plus grands jeux de données.",
                tags: ["Entraînement parallèle", "Données volumineuses", "Optimisation distribuée"],
                color: "#00B0F0"
              },
              {
                title: "Calcul scientifique",
                desc: "Simulez des phénomènes complexes et analysez d'importants volumes de données avec une puissance de calcul distribuée.",
                tags: ["Modélisation", "Simulations", "Analyses statistiques"],
                color: "#00D4FF"
              },
              {
                title: "Traitement de données",
                desc: "Transformez et analysez de grandes quantités de données réparties sur plusieurs machines pour des résultats plus rapides.",
                tags: ["ETL distribué", "Big Data", "Analyses batch"],
                color: "#00B0F0"
              },
              {
                title: "Optimisation",
                desc: "Résolvez des problèmes d'optimisation complexes en parallélisant les calculs sur plusieurs ressources.",
                tags: ["Algorithmes génétiques", "Recherche de solutions", "Optimisation contrainte"],
                color: "#00D4FF"
              }
            ].map((app, idx) => (
              <div key={idx} className="flex gap-6 p-6 rounded-2xl transition-all duration-300"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.6) 0%, rgba(0, 20, 64, 0.6) 100%)',
                  border: '2px solid rgba(0, 180, 240, 0.3)',
                  backdropFilter: 'blur(10px)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-5px)';
                  e.currentTarget.style.borderColor = app.color;
                  e.currentTarget.style.boxShadow = `0 12px 40px ${app.color}40`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.borderColor = 'rgba(0, 180, 240, 0.3)';
                  e.currentTarget.style.boxShadow = 'none';
                }}>
                
                <div className="flex-shrink-0 w-16 h-16 rounded-xl flex items-center justify-center"
                  style={{
                    background: `linear-gradient(135deg, ${app.color}20 0%, ${app.color}10 100%)`,
                    border: `2px solid ${app.color}40`,
                  }}>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    {idx === 0 && (
                      <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" stroke={app.color} strokeWidth="2"/>
                    )}
                    {idx === 1 && (
                      <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" stroke={app.color} strokeWidth="2"/>
                    )}
                    {idx === 2 && (
                      <path d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" stroke={app.color} strokeWidth="2"/>
                    )}
                    {idx === 3 && (
                      <path d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" stroke={app.color} strokeWidth="2"/>
                    )}
                  </svg>
                </div>

                <div className="flex-1">
                  <h3 className="text-xl font-bold mb-2" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>
                    {app.title}
                  </h3>
                  <p className="mb-4" style={{ color: '#00B0F0', fontSize: '14px' }}>
                    {app.desc}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {app.tags.map((tag, tidx) => (
                      <span key={tidx} className="px-3 py-1 rounded-full text-xs font-medium"
                        style={{
                          background: `${app.color}20`,
                          color: app.color,
                          border: `1px solid ${app.color}40`,
                        }}>
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0"
          style={{
            background: 'radial-gradient(circle at center, rgba(0, 212, 255, 0.1) 0%, transparent 70%)',
          }} />
        
        <div className="container mx-auto px-6 text-center relative z-10">
          <div className="max-w-3xl mx-auto p-12 rounded-3xl"
            style={{
              background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.8) 0%, rgba(0, 20, 64, 0.8) 100%)',
              border: '2px solid rgba(0, 180, 240, 0.4)',
              backdropFilter: 'blur(20px)',
              boxShadow: '0 16px 48px rgba(0, 32, 96, 0.6)',
            }}>
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl mb-6"
              style={{
                background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.25) 0%, rgba(0, 180, 240, 0.15) 100%)',
                border: '2px solid rgba(0, 212, 255, 0.4)',
                animation: 'pulse 3s ease-in-out infinite',
              }}>
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" fill="#00D4FF"/>
              </svg>
            </div>
            <h2 className="text-4xl font-bold mb-6" style={{ color: '#FFFFFF', letterSpacing: '0.5px' }}>
              Prêt à optimiser vos calculs ?
            </h2>
            <p className="text-xl mb-10" style={{ color: '#00B0F0' }}>
              Rejoignez notre plateforme et transformez votre approche des tâches de calcul intensif.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/register" 
                className="flex items-center justify-center gap-2 px-10 py-4 rounded-xl font-bold transition-all duration-300"
                style={{
                  background: 'linear-gradient(135deg, #00B0F0 0%, #00D4FF 100%)',
                  color: '#FFFFFF',
                  border: '2px solid rgba(0, 212, 255, 0.4)',
                  boxShadow: '0 8px 32px rgba(0, 180, 240, 0.4)',
                  letterSpacing: '0.5px',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-3px) scale(1.05)';
                  e.currentTarget.style.boxShadow = '0 12px 40px rgba(0, 212, 255, 0.6)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0) scale(1)';
                  e.currentTarget.style.boxShadow = '0 8px 32px rgba(0, 180, 240, 0.4)';
                }}>
                Commencer gratuitement
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M13 10V3L4 14h7v7l9-11h-7z" fill="currentColor"/>
                </svg>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative border-t"
        style={{
          background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.9) 0%, rgba(0, 20, 64, 0.9) 100%)',
          borderColor: 'rgba(0, 180, 240, 0.2)',
        }}>
        <div className="container mx-auto px-6 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="md:col-span-2">
              <div className="flex items-center gap-3 mb-4">
                <div className="h-10 w-10 rounded-xl flex items-center justify-center"
                  style={{
                    background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.25) 0%, rgba(0, 180, 240, 0.15) 100%)',
                    border: '2px solid rgba(0, 212, 255, 0.4)',
                  }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="3" stroke="#00D4FF" strokeWidth="2"/>
                    <circle cx="12" cy="5" r="2" fill="#00D4FF"/>
                    <circle cx="12" cy="19" r="2" fill="#00D4FF"/>
                    <circle cx="5" cy="12" r="2" fill="#00B0F0"/>
                    <circle cx="19" cy="12" r="2" fill="#00B0F0"/>
                  </svg>
                </div>
                <span className="text-xl font-bold" style={{ color: '#FFFFFF', letterSpacing: '0.5px' }}>
                  VolunSys-UY1
                </span>
              </div>
              <p className="mb-4" style={{ color: '#00B0F0', fontSize: '14px' }}>
                Une plateforme distribuée pour l'orchestration de workflows de calcul intensif développée dans le cadre d'un projet universitaire.
              </p>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 14l9-5-9-5-9 5 9 5z" stroke="#00D4FF" strokeWidth="2"/>
                    <path d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" stroke="#00D4FF" strokeWidth="2"/>
                  </svg>
                  <span style={{ color: '#FFFFFF', fontSize: '13px', fontWeight: 500 }}>
                    Projet Master I - Université de Yaoundé
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" stroke="#00D4FF" strokeWidth="2"/>
                  </svg>
                  <span style={{ color: '#00D4FF', fontSize: '13px', fontWeight: 600 }}>
                    Encadré par Dr Hamza ADAMOU
                  </span>
                </div>
              </div>
              <p className="mt-4 text-xs" style={{ color: '#00B0F0', opacity: 0.7 }}>
                © {new Date().getFullYear()} VolunSys-UY1. Tous droits réservés.
              </p>
            </div>
            
            <div>
              <h3 className="text-lg font-bold mb-4" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>
                Liens rapides
              </h3>
              <ul className="space-y-3">
                {['Accueil', 'Comment ça marche', 'Applications', 'Fonctionnalités'].map((link, idx) => (
                  <li key={idx}>
                    <a href="#" className="text-sm transition-all duration-300"
                      style={{ color: '#00B0F0' }}
                      onMouseEnter={(e) => { e.currentTarget.style.color = '#00D4FF'; e.currentTarget.style.paddingLeft = '8px'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.color = '#00B0F0'; e.currentTarget.style.paddingLeft = '0'; }}>
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-bold mb-4" style={{ color: '#FFFFFF', letterSpacing: '0.3px' }}>
                Accès
              </h3>
              <ul className="space-y-3">
                <li>
                  <Link href="/login" className="text-sm transition-all duration-300"
                    style={{ color: '#00B0F0' }}
                    onMouseEnter={(e) => { e.currentTarget.style.color = '#00D4FF'; e.currentTarget.style.paddingLeft = '8px'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.color = '#00B0F0'; e.currentTarget.style.paddingLeft = '0'; }}>
                    Connexion
                  </Link>
                </li>
                <li>
                  <Link href="/register" className="text-sm transition-all duration-300"
                    style={{ color: '#00B0F0' }}
                    onMouseEnter={(e) => { e.currentTarget.style.color = '#00D4FF'; e.currentTarget.style.paddingLeft = '8px'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.color = '#00B0F0'; e.currentTarget.style.paddingLeft = '0'; }}>
                    Inscription
                  </Link>
                </li>
              </ul>
              
              <div className="mt-6 p-4 rounded-xl"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 180, 240, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%)',
                  border: '1px solid rgba(0, 180, 240, 0.2)',
                }}>
                <div className="flex items-center gap-2 mb-2">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" stroke="#00D4FF" strokeWidth="2"/>
                  </svg>
                  <span style={{ color: '#00D4FF', fontSize: '12px', fontWeight: 600 }}>
                    Communauté Active
                  </span>
                </div>
                <p style={{ color: '#FFFFFF', fontSize: '11px' }}>
                  Rejoignez des centaines de chercheurs et développeurs
                </p>
              </div>
            </div>
          </div>
          
          <div className="mt-8 pt-8 text-center"
            style={{
              borderTop: '1px solid rgba(0, 180, 240, 0.2)',
            }}>
            <p style={{ color: '#00B0F0', fontSize: '12px', letterSpacing: '0.5px' }}>
              La puissance collective au service du calcul scientifique
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}