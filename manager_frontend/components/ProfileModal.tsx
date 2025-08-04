'use client';
import { Fragment, useState } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { authService } from '@/lib/api';
import { useRouter } from 'next/navigation';

export function ProfileModal() {
  const [isOpen, setIsOpen] = useState(false);
  const router = useRouter();

  const user = authService.getCurrentUser();

  const closeModal = () => setIsOpen(false);
  const openModal = () => setIsOpen(true);

  const handleLogout = () => {
    authService.logout();
    router.push('/login');
  };

  if (!user) return null;

  return (
    <>
      <button onClick={openModal} className="group relative inline-flex items-center gap-3 px-4 py-2.5 bg-gradient-to-r from-slate-800 via-slate-900 to-slate-800 hover:from-slate-700 hover:via-slate-800 hover:to-slate-700 text-white font-medium rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 border border-slate-700/50 hover:border-slate-600/50">
        <div className="relative w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-inner">
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-4 w-4 text-white" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth="2" 
              d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
            />
          </svg>
          {/* Indicator de statut */}
          <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-green-400 border-2 border-slate-800 rounded-full"></div>
        </div>
        
        <span className="text-sm font-medium">{user.username}</span>
        
        {/* Icône dropdown */}
        <svg 
          className="w-3 h-3 text-slate-400 group-hover:text-white transition-colors duration-200" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Modale moderne */}
      <Transition appear show={isOpen} as={Fragment}>
        <Dialog as="div" className="relative z-50" onClose={closeModal}>
          {/* Backdrop avec effet blur */}
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" />
          </Transition.Child>

          <div className="fixed inset-0 overflow-y-auto">
            <div className="flex min-h-full items-center justify-center p-4">
              <Transition.Child
                as={Fragment}
                enter="ease-out duration-300"
                enterFrom="opacity-0 scale-95 translate-y-4"
                enterTo="opacity-100 scale-100 translate-y-0"
                leave="ease-in duration-200"
                leaveFrom="opacity-100 scale-100 translate-y-0"
                leaveTo="opacity-0 scale-95 translate-y-4"
              >
                <Dialog.Panel className="w-full max-w-md transform overflow-hidden rounded-2xl bg-white/95 backdrop-blur-xl border border-gray-200/50 shadow-2xl transition-all">
                  {/* Header avec gradient */}
                  <div className="relative bg-gradient-to-br from-slate-800 via-slate-900 to-slate-800 px-6 py-8 text-white">
                    {/* Pattern décoratif */}
\                    
                    <div className="relative flex items-center gap-4">
                      {/* Grand avatar */}
                      <div className="w-16 h-16 bg-gradient-to-br from-blue-400 via-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg">
                        <svg 
                          xmlns="http://www.w3.org/2000/svg" 
                          className="h-8 w-8 text-white" 
                          fill="none" 
                          viewBox="0 0 24 24" 
                          stroke="currentColor"
                        >
                          <path 
                            strokeLinecap="round" 
                            strokeLinejoin="round" 
                            strokeWidth="1.5" 
                            d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                          />
                        </svg>
                      </div>
                      
                      <div>
                        <Dialog.Title as="h3" className="text-xl font-bold">
                          Profil utilisateur
                        </Dialog.Title>
                        <p className="text-slate-300 text-sm mt-1">Gérez votre compte</p>
                      </div>
                    </div>

                    {/* Bouton fermer */}
                    <button
                      onClick={closeModal}
                      className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors duration-200"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>

                  {/* Contenu */}
                  <div className="p-6 space-y-6">
                    {/* Informations utilisateur */}
                    <div className="space-y-4">
                      <div className="bg-gray-50/80 rounded-xl p-4 border border-gray-100">
                        <div className="flex items-center gap-3 mb-2">
                          <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                            <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                          </div>
                          <div>
                            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Nom d'utilisateur</p>
                            <p className="text-gray-900 font-semibold">{user.username}</p>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-50/80 rounded-xl p-4 border border-gray-100">
                        <div className="flex items-center gap-3 mb-2">
                          <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                            <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                            </svg>
                          </div>
                          <div>
                            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Email</p>
                            <p className="text-gray-900 font-semibold">{user.email}</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-3 pt-2">
                      <button
                        type="button"
                        className="flex-1 px-4 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-xl transition-colors duration-200"
                        onClick={closeModal}
                      >
                        Fermer
                      </button>
                      <button
                        type="button"
                        className="flex-1 px-4 py-2.5 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white font-medium rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center gap-2"
                        onClick={handleLogout}
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                        </svg>
                        Déconnexion
                      </button>
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </Dialog>
      </Transition>
    </>
  );
}