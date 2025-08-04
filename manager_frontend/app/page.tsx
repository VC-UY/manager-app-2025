'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useState } from 'react';


export default function Home() {
  const [activeFeature, setActiveFeature] = useState(0);
  
  const features = [
    {
      title: "Parallélisation Intelligente",
      description: "Notre système analyse automatiquement votre workflow et identifie les parties qui peuvent être exécutées en parallèle pour maximiser l'efficacité.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      )
    },
    {
      title: "Tolérance aux Pannes",
      description: "Le système détecte automatiquement les défaillances et réattribue les tâches à d'autres volontaires, garantissant la robustesse et la fiabilité de vos calculs.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
      )
    },
    {
      title: "Surveillance en Temps Réel",
      description: "Visualisez l'avancement de vos tâches en temps réel avec des statistiques détaillées et des indicateurs de performance pour optimiser vos workflows.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    }
  ];

  return (
    <main className="min-h-screen">
      {/* Navbar */}
      <nav className="bg-white shadow-sm py-4">
        <div className="container mx-auto px-6 flex justify-between items-center">
          <div className="flex items-center">
            <div className="h-10 w-10 rounded-full bg-gradient-to-r from-blue-600 to-indigo-700 flex items-center justify-center text-white font-bold text-xl mr-3">
              CV
            </div>
            <span className="text-xl font-bold text-gray-800">Calcul Volontaire</span>
          </div>
          <div className="hidden md:flex space-x-6">
            <a href="#how-it-works" className="text-gray-600 hover:text-blue-600 transition-colors">Comment ça marche</a>
            <a href="#applications" className="text-gray-600 hover:text-blue-600 transition-colors">Applications</a>
            <a href="#features" className="text-gray-600 hover:text-blue-600 transition-colors">Fonctionnalités</a>
          </div>
          <div className="flex items-center space-x-4">
            <Link 
              href="/login" 
              className="text-gray-800 hover:text-blue-600 transition-colors"
            >
              Connexion
            </Link>
            <Link 
              href="/register" 
              className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg text-sm font-medium transition-colors"
            >
              S'inscrire
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative bg-gradient-to-r from-blue-600 to-indigo-700 text-white overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden opacity-10">
          <div className="absolute -top-10 -left-10 w-40 h-40 bg-white rounded-full"></div>
          <div className="absolute top-1/3 -right-10 w-60 h-60 bg-white rounded-full"></div>
          <div className="absolute -bottom-20 left-1/4 w-80 h-80 bg-white rounded-full"></div>
        </div>
        
        <div className="container mx-auto px-6 py-24 md:py-32 relative z-10">
          <div className="flex flex-col md:flex-row items-center justify-between">
            <div className="md:w-1/2 text-center md:text-left mb-10 md:mb-0">
              <h1 className="text-5xl md:text-6xl font-extrabold mb-6 leading-tight">
                Calcul Distribué<br />
                <span className="text-blue-200">Simple et Puissant</span>
              </h1>
              <p className="text-xl md:text-2xl mb-10 text-blue-100 max-w-xl">
                Une plateforme innovante qui orchestre intelligemment vos workflows de calcul intensif sur un réseau de volontaires.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center md:justify-start">
                <Link 
                  href="/register" 
                  className="bg-white text-blue-700 hover:bg-blue-50 py-3 px-8 rounded-lg font-semibold shadow-lg transition-all flex items-center justify-center"
                >
                  <span>Commencer gratuitement</span>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </Link>
                <Link 
                  href="#how-it-works" 
                  className="bg-transparent hover:bg-blue-600 text-white border-2 border-white py-3 px-8 rounded-lg font-semibold transition-all flex items-center justify-center"
                >
                  <span>En savoir plus</span>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v3.586L7.707 9.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 10.586V7z" clipRule="evenodd" />
                  </svg>
                </Link>
              </div>
            </div>
            <div className="md:w-1/2 relative">
              {/* Hero illustration */}
              <div className="relative h-64 md:h-96 w-full">
                <div className="absolute inset-0 bg-blue-800 bg-opacity-30 rounded-2xl backdrop-blur-sm shadow-xl p-6">
                  <div className="h-full flex flex-col">
                    <div className="flex items-center mb-4">
                      <div className="h-3 w-3 rounded-full bg-red-500 mr-2"></div>
                      <div className="h-3 w-3 rounded-full bg-yellow-500 mr-2"></div>
                      <div className="h-3 w-3 rounded-full bg-green-500"></div>
                      <div className="ml-4 text-white opacity-80 text-sm">WorkflowManager.js</div>
                    </div>
                    <div className="flex-grow overflow-hidden font-mono text-xs text-blue-100 opacity-90 leading-relaxed">
                      <div>&gt; Initializing workflow...</div>
                      <div className="mt-2">&gt; Analyzing computational requirements</div>
                      <div className="mt-2">&gt; Splitting into 8 parallel tasks</div>
                      <div className="mt-2">&gt; Finding available volunteers...</div>
                      <div className="mt-2">&gt; Assigning tasks to optimal nodes</div>
                      <div className="mt-2 flex items-center">
                        &gt; Processing... <span className="inline-block ml-2 h-2 w-2 bg-blue-200 rounded-full animate-pulse"></span>
                      </div>
                      <div className="mt-2">
                        &gt; Task progress:
                        <div className="mt-1 h-4 bg-blue-800 rounded overflow-hidden">
                          <div className="h-full bg-blue-300 w-2/3 animate-pulse-slow"></div>
                        </div>
                      </div>
                      <div className="mt-2">&gt; <span className="text-green-300">3</span> tasks completed | <span className="text-yellow-300">4</span> tasks running | <span className="text-blue-300">1</span> tasks pending</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Wave separator */}
        <div className="absolute bottom-0 left-0 right-0 h-16 -mb-1">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" className="w-full h-full">
            <path fill="#f9fafb" fillOpacity="1" d="M0,160L48,138.7C96,117,192,75,288,74.7C384,75,480,117,576,144C672,171,768,181,864,170.7C960,160,1056,128,1152,117.3C1248,107,1344,117,1392,122.7L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
          </svg>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 bg-gray-50">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Comment ça fonctionne</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Une approche en trois étapes pour transformer vos calculs complexes en tâches distribuées et efficaces.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            {/* Connection line between steps */}
            <div className="hidden md:block absolute h-0.5 bg-blue-200 top-1/4 left-[25%] right-[25%] transform -translate-y-1/2"></div>

            <div className="bg-white p-8 rounded-xl shadow-lg relative transition-all hover:shadow-xl hover:-translate-y-1">
              <div className="h-14 w-14 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mb-6 absolute -top-7 left-1/2 transform -translate-x-1/2 shadow-md">
                <span className="text-2xl font-bold">1</span>
              </div>
              <div className="pt-8">
                <h3 className="text-xl font-semibold mb-4 text-center">Créez votre workflow</h3>
                <p className="text-gray-600 mb-6">
                  Définissez votre workflow de calcul intensif, importez vos données et configurez les paramètres d'exécution selon vos besoins spécifiques.
                </p>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <ul className="text-sm text-gray-700 space-y-2">
                    <li className="flex items-start">
                      <svg className="h-5 w-5 text-blue-500 mr-2 flex-shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Interface intuitive pour définir vos tâches
                    </li>
                    <li className="flex items-start">
                      <svg className="h-5 w-5 text-blue-500 mr-2 flex-shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Importation simple des données et scripts
                    </li>
                    <li className="flex items-start">
                      <svg className="h-5 w-5 text-blue-500 mr-2 flex-shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Estimation automatique des ressources
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-lg relative transition-all hover:shadow-xl hover:-translate-y-1">
              <div className="h-14 w-14 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center mb-6 absolute -top-7 left-1/2 transform -translate-x-1/2 shadow-md">
                <span className="text-2xl font-bold">2</span>
              </div>
              <div className="pt-8">
                <h3 className="text-xl font-semibold mb-4 text-center">Orchestration automatique</h3>
                <p className="text-gray-600 mb-6">
                  Notre système analyse vos tâches, les décompose intelligemment et les distribue aux nœuds de calcul volontaires optimaux disponibles.
                </p>
                <div className="bg-indigo-50 p-4 rounded-lg">
                  <ul className="text-sm text-gray-700 space-y-2">
                    <li className="flex items-start">
                      <svg className="h-5 w-5 text-indigo-500 mr-2 flex-shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Découpage intelligent en sous-tâches
                    </li>
                    <li className="flex items-start">
                      <svg className="h-5 w-5 text-indigo-500 mr-2 flex-shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Attribution optimale aux volontaires
                    </li>
                    <li className="flex items-start">
                      <svg className="h-5 w-5 text-indigo-500 mr-2 flex-shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Gestion automatique des dépendances
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-lg relative transition-all hover:shadow-xl hover:-translate-y-1">
              <div className="h-14 w-14 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center mb-6 absolute -top-7 left-1/2 transform -translate-x-1/2 shadow-md">
                <span className="text-2xl font-bold">3</span>
              </div>
              <div className="pt-8">
                <h3 className="text-xl font-semibold mb-4 text-center">Résultats consolidés</h3>
                <p className="text-gray-600 mb-6">
                  Suivez en temps réel l'avancement de chaque tâche et récupérez les résultats agrégés et validés une fois le traitement terminé.
                </p>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <ul className="text-sm text-gray-700 space-y-2">
                    <li className="flex items-start">
                      <svg className="h-5 w-5 text-purple-500 mr-2 flex-shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Visualisation en temps réel
                    </li>
                    <li className="flex items-start">
                      <svg className="h-5 w-5 text-purple-500 mr-2 flex-shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Agrégation intelligente des résultats
                    </li>
                    <li className="flex items-start">
                      <svg className="h-5 w-5 text-purple-500 mr-2 flex-shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Exportation dans différents formats
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-white">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Fonctionnalités avancées</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Des outils puissants pour gérer et optimiser vos calculs distribués.
            </p>
          </div>
          
          <div className="flex flex-col md:flex-row">
            {/* Feature tabs */}
            <div className="md:w-1/3 mb-8 md:mb-0 md:pr-8">
              {features.map((feature, index) => (
                <div 
                  key={index} 
                  className={`p-6 mb-4 rounded-lg cursor-pointer transition-all ${
                    activeFeature === index 
                      ? 'bg-blue-600 text-white shadow-lg' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                  onClick={() => setActiveFeature(index)}
                >
                  <div className="flex items-center">
                    <div className={`flex-shrink-0 mr-4 ${
                      activeFeature === index ? 'text-white' : 'text-blue-600'
                    }`}>
                      {feature.icon}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold">{feature.title}</h3>
                      <p className={`text-sm mt-1 ${
                        activeFeature === index ? 'text-blue-100' : 'text-gray-500'
                      }`}>
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Feature visualization */}
            <div className="md:w-2/3 bg-gray-50 rounded-2xl overflow-hidden shadow-lg">
              <div className="h-96 p-8 relative">
                {activeFeature === 0 && (
                  <div className="h-full flex items-center justify-center">
                    <div className="w-full max-w-md">
                      <div className="text-center mb-6">
                        <span className="inline-block p-3 bg-blue-100 text-blue-600 rounded-full">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                          </svg>
                        </span>
                      </div>
                      
                      <div className="bg-white rounded-lg shadow-md p-4 mb-4">
                        <div className="h-4 bg-gray-100 w-full rounded"></div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-white rounded-lg shadow-md p-4">
                          <div className="h-4 bg-gray-100 w-3/4 rounded mb-2"></div>
                          <div className="h-4 bg-gray-100 w-1/2 rounded"></div>
                        </div>
                        <div className="bg-white rounded-lg shadow-md p-4">
                          <div className="h-4 bg-gray-100 w-3/4 rounded mb-2"></div>
                          <div className="h-4 bg-gray-100 w-1/2 rounded"></div>
                        </div>
                        <div className="bg-white rounded-lg shadow-md p-4">
                          <div className="h-4 bg-gray-100 w-3/4 rounded mb-2"></div>
                          <div className="h-4 bg-gray-100 w-1/2 rounded"></div>
                        </div>
                        <div className="bg-white rounded-lg shadow-md p-4">
                          <div className="h-4 bg-gray-100 w-3/4 rounded mb-2"></div>
                          <div className="h-4 bg-gray-100 w-1/2 rounded"></div>
                        </div>
                      </div>
                      
                      <svg className="absolute bottom-8 right-8 text-blue-500 opacity-10 h-40 w-40">
                        <path d="M13 10V3L4 14h7v7l9-11h-7z" fill="currentColor" />
                      </svg>
                    </div>
                  </div>
                )}
                
                {activeFeature === 1 && (
                  <div className="h-full flex items-center justify-center">
                    <div className="w-full max-w-md">
                      <div className="text-center mb-6">
                        <span className="inline-block p-3 bg-blue-100 text-blue-600 rounded-full">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                          </svg>
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="bg-white rounded-lg shadow-md p-4 flex flex-col items-center">
                          <div className="h-8 w-8 rounded-full bg-green-100 text-green-500 flex items-center justify-center mb-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                          </div>
                          <div className="text-sm font-medium text-gray-700">Node 1</div>
                          <div className="text-xs text-gray-500">Actif</div>
                        </div>
                        <div className="bg-white rounded-lg shadow-md p-4 flex flex-col items-center">
                          <div className="h-8 w-8 rounded-full bg-yellow-100 text-yellow-500 flex items-center justify-center mb-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                          </div>
                          <div className="text-sm font-medium text-gray-700">Node 2</div>
                          <div className="text-xs text-yellow-500">Instable</div>
                        </div>
                        <div className="bg-white rounded-lg shadow-md p-4 flex flex-col items-center">
                          <div className="h-8 w-8 rounded-full bg-red-100 text-red-500 flex items-center justify-center mb-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                          </div>
                          <div className="text-sm font-medium text-gray-700">Node 3</div>
                          <div className="text-xs text-red-500">Déconnecté</div>
                        </div>
                      </div>
                      
                      <div className="bg-white rounded-lg shadow-md p-4 relative">
                        <div className="flex mb-2">
                          <div className="flex-grow">
                            <div className="h-4 bg-gray-100 w-3/4 rounded"></div>
                          </div>
                          <div className="flex-shrink-0 ml-2">
                            <div className="h-4 w-20 bg-green-100 rounded"></div>
                          </div>
                        </div>
                        
                        <div className="h-20 bg-gray-50 rounded-lg flex items-center justify-center p-2">
                          <div className="text-xs text-center text-gray-500">
                            <div className="mb-1 font-semibold">Tâche redistribuée</div>
                            <div>Node 2 → Node 1</div>
                          </div>
                          <svg className="absolute bottom-6 right-6 h-6 w-6 text-green-500">
                            <path d="M5 3l3.5 3L12 3l-3.5 3L12 9 5 3zm0 6l3.5 3L12 9l-3.5 3L12 15l-7-6z" fill="currentColor" />
                          </svg>
                        </div>
                      </div>
                      
                      <svg className="absolute bottom-8 right-8 text-blue-500 opacity-10 h-40 w-40">
                        <path d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" stroke="currentColor" strokeWidth="4" fill="none" />
                      </svg>
                    </div>
                  </div>
                )}
                
                {activeFeature === 2 && (
                  <div className="h-full flex items-center justify-center">
                    <div className="w-full max-w-md">
                      <div className="text-center mb-6">
                        <span className="inline-block p-3 bg-blue-100 text-blue-600 rounded-full">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                        </span>
                      </div>
                      
                      <div className="bg-white rounded-lg shadow-md p-4 mb-4">
                        <div className="flex justify-between items-center mb-3">
                          <div className="h-4 bg-gray-100 w-1/3 rounded"></div>
                          <div className="bg-blue-100 text-blue-600 text-xs px-2 py-1 rounded-full">
                            Live
                          </div>
                        </div>
                        
                        <div className="h-24 bg-gray-50 rounded-lg mb-3 relative overflow-hidden">
                          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-r from-green-400 to-blue-500 h-16"></div>
                          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-r from-green-400 to-blue-500 h-8 opacity-20"></div>
                          
                          <div className="absolute top-0 left-0 right-0 bottom-0 flex items-center justify-center">
                            <div className="text-sm font-semibold text-gray-700">Progression globale: 67%</div>
                          </div>
                        </div>
                        
                        <div className="flex justify-between text-xs text-gray-500">
                          <div>0%</div>
                          <div>100%</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-white rounded-lg shadow-md p-3">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center">
                              <div className="h-3 w-3 rounded-full bg-green-500 mr-2"></div>
                              <div className="text-xs font-medium text-gray-700">Tâche 1</div>
                            </div>
                            <div className="text-xs font-medium text-green-600">100%</div>
                          </div>
                          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div className="h-full bg-green-500 w-full"></div>
                          </div>
                        </div>
                        
                        <div className="bg-white rounded-lg shadow-md p-3">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center">
                              <div className="h-3 w-3 rounded-full bg-blue-500 mr-2"></div>
                              <div className="text-xs font-medium text-gray-700">Tâche 2</div>
                            </div>
                            <div className="text-xs font-medium text-blue-600">75%</div>
                          </div>
                          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 w-3/4"></div>
                          </div>
                        </div>
                        
                        <div className="bg-white rounded-lg shadow-md p-3">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center">
                              <div className="h-3 w-3 rounded-full bg-yellow-500 mr-2"></div>
                              <div className="text-xs font-medium text-gray-700">Tâche 3</div>
                            </div>
                            <div className="text-xs font-medium text-yellow-600">40%</div>
                          </div>
                          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div className="h-full bg-yellow-500 w-2/5"></div>
                          </div>
                        </div>
                        
                        <div className="bg-white rounded-lg shadow-md p-3">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center">
                              <div className="h-3 w-3 rounded-full bg-purple-500 mr-2"></div>
                              <div className="text-xs font-medium text-gray-700">Tâche 4</div>
                            </div>
                            <div className="text-xs font-medium text-purple-600">60%</div>
                          </div>
                          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div className="h-full bg-purple-500 w-3/5"></div>
                          </div>
                        </div>
                      </div>
                      
                      <svg className="absolute bottom-8 right-8 text-blue-500 opacity-10 h-40 w-40">
                        <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" stroke="currentColor" strokeWidth="4" fill="none" />
                      </svg>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Applications Section */}
      <section id="applications" className="py-20 bg-gray-50">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Applications</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Notre plateforme s'adapte à diverses applications de calcul intensif.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            <div className="flex flex-col md:flex-row gap-6 bg-white p-6 rounded-xl shadow-md transition-all hover:shadow-lg hover:-translate-y-1">
              <div className="flex-shrink-0 h-16 w-16 bg-green-100 text-green-600 rounded-lg flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Machine Learning</h3>
                <p className="text-gray-600 mb-4">
                  Distribuez l'entraînement de vos modèles sur plusieurs nœuds pour accélérer la convergence et traiter de plus grands jeux de données.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="bg-green-50 text-green-600 text-xs px-2 py-1 rounded-full">Entraînement parallèle</span>
                  <span className="bg-green-50 text-green-600 text-xs px-2 py-1 rounded-full">Données volumineuses</span>
                  <span className="bg-green-50 text-green-600 text-xs px-2 py-1 rounded-full">Optimisation distribuée</span>
                </div>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-6 bg-white p-6 rounded-xl shadow-md transition-all hover:shadow-lg hover:-translate-y-1">
              <div className="flex-shrink-0 h-16 w-16 bg-purple-100 text-purple-600 rounded-lg flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Calcul scientifique</h3>
                <p className="text-gray-600 mb-4">
                  Simulez des phénomènes complexes et analysez d'importants volumes de données avec une puissance de calcul distribuée.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="bg-purple-50 text-purple-600 text-xs px-2 py-1 rounded-full">Modélisation</span>
                  <span className="bg-purple-50 text-purple-600 text-xs px-2 py-1 rounded-full">Simulations</span>
                  <span className="bg-purple-50 text-purple-600 text-xs px-2 py-1 rounded-full">Analyses statistiques</span>
                </div>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-6 bg-white p-6 rounded-xl shadow-md transition-all hover:shadow-lg hover:-translate-y-1">
              <div className="flex-shrink-0 h-16 w-16 bg-yellow-100 text-yellow-600 rounded-lg flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Traitement de données</h3>
                <p className="text-gray-600 mb-4">
                  Transformez et analysez de grandes quantités de données réparties sur plusieurs machines pour des résultats plus rapides.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="bg-yellow-50 text-yellow-600 text-xs px-2 py-1 rounded-full">ETL distribué</span>
                  <span className="bg-yellow-50 text-yellow-600 text-xs px-2 py-1 rounded-full">Big Data</span>
                  <span className="bg-yellow-50 text-yellow-600 text-xs px-2 py-1 rounded-full">Analyses batch</span>
                </div>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-6 bg-white p-6 rounded-xl shadow-md transition-all hover:shadow-lg hover:-translate-y-1">
              <div className="flex-shrink-0 h-16 w-16 bg-red-100 text-red-600 rounded-lg flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Optimisation</h3>
                <p className="text-gray-600 mb-4">
                  Résolvez des problèmes d'optimisation complexes en parallélisant les calculs sur plusieurs ressources.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="bg-red-50 text-red-600 text-xs px-2 py-1 rounded-full">Algorithmes génétiques</span>
                  <span className="bg-red-50 text-red-600 text-xs px-2 py-1 rounded-full">Recherche de solutions</span>
                  <span className="bg-red-50 text-red-600 text-xs px-2 py-1 rounded-full">Optimisation contrainte</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-blue-50 rounded-xl p-8 text-center transition-all hover:shadow-md">
              <div className="text-4xl font-bold text-blue-600 mb-2">+ 300%</div>
              <p className="text-gray-700">Augmentation des performances</p>
            </div>
            <div className="bg-green-50 rounded-xl p-8 text-center transition-all hover:shadow-md">
              <div className="text-4xl font-bold text-green-600 mb-2">- 65%</div>
              <p className="text-gray-700">Réduction du temps de calcul</p>
            </div>
            <div className="bg-purple-50 rounded-xl p-8 text-center transition-all hover:shadow-md">
              <div className="text-4xl font-bold text-purple-600 mb-2">∞</div>
              <p className="text-gray-700">Évolutivité illimitée</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-indigo-700 text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold mb-6">Prêt à optimiser vos calculs ?</h2>
          <p className="text-xl mb-10 max-w-3xl mx-auto">
            Rejoignez notre plateforme et transformez votre approche des tâches de calcul intensif.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link 
              href="/register" 
              className="bg-white text-blue-700 hover:bg-blue-50 py-4 px-10 rounded-lg font-semibold shadow-lg transition-all flex items-center justify-center"
            >
              <span>Commencer gratuitement</span>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </Link>
            <Link 
              href="#how-it-works" 
              className="bg-transparent hover:bg-blue-600 text-white border-2 border-white py-4 px-10 rounded-lg font-semibold transition-all flex items-center justify-center"
            >
              <span>En savoir plus</span>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v3.586L7.707 9.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 10.586V7z" clipRule="evenodd" />
              </svg>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-300 py-12">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="md:col-span-2">
              <div className="flex items-center mb-4">
                <div className="h-10 w-10 rounded-full bg-gradient-to-r from-blue-600 to-indigo-700 flex items-center justify-center text-white font-bold text-xl mr-3">
                  CV
                </div>
                <span className="text-xl font-bold text-white">Calcul Volontaire</span>
              </div>
              <p className="mb-4 text-gray-400">
                Une plateforme distribuée pour l'orchestration de workflows de calcul intensif développée dans le cadre d'un projet universitaire.
              </p>
              <p className="text-sm">
                © 2025 Master I - Université de Yaoundé. Tous droits réservés.
              </p>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4 text-white">Liens rapides</h3>
              <ul className="space-y-2">
                <li><a href="#" className="text-gray-400 hover:text-white transition-colors">Accueil</a></li>
                <li><a href="#how-it-works" className="text-gray-400 hover:text-white transition-colors">Comment ça marche</a></li>
                <li><a href="#applications" className="text-gray-400 hover:text-white transition-colors">Applications</a></li>
                <li><a href="#features" className="text-gray-400 hover:text-white transition-colors">Fonctionnalités</a></li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4 text-white">Projet</h3>
              <ul className="space-y-2">
                <li><a href="/login" className="text-gray-400 hover:text-white transition-colors">Connexion</a></li>
                <li><a href="/register" className="text-gray-400 hover:text-white transition-colors">Inscription</a></li>
                <li><span className="text-gray-400">Un projet encadré par</span></li>
                <li><span className="text-gray-300 font-medium">Dr Hamza ADAMOU</span></li>
              </ul>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}