'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { authService } from '@/lib/api';

interface User {
  id: string;
  first_name: string;
  last_name: string;
  email: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (userData: { first_name: string; last_name: string; email: string; password: string; password2: string }) => Promise<void>;
  logout: () => Promise<void>;
  isAuthenticated: boolean;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  // Vérifier l'authentification au chargement
  useEffect(() => {
    const checkAuth = () => {
      try {
        if (authService.isAuthenticated()) {
          const currentUser = authService.getCurrentUser();
          if (currentUser) {
            console.log("[Auth] Utilisateur chargé depuis le stockage local:", currentUser.email);
            setUser(currentUser);
          }
        } else {
          console.log("[Auth] Aucun utilisateur authentifié");
        }
      } catch (error) {
        console.error('[Auth] Erreur de vérification d\'authentification:', error);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  // Effacer les erreurs
  const clearError = () => setError(null);

  // Fonction de connexion avec meilleure gestion d'erreur
  const login = async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('[Auth] Tentative de connexion avec:', email);
      
      const data = await authService.login({ email, password });
      console.log('[Auth] Connexion réussie:', data);
      
      if (data && data.user) {
        setUser(data.user);
        
        // Ajouter un délai avant la redirection
        setTimeout(() => {
          console.log('[Auth] Redirection vers les workflows');
          router.push('/workflows');
        }, 300);
      } else {
        throw new Error('Réponse du serveur invalide');
      }
    } catch (error: any) {
      console.error('[Auth] Erreur de connexion:', error);
      
      // Extraction optimisée du message d'erreur
      let errorMessage = 'Une erreur est survenue lors de la connexion';
      
      if (error?.error) {
        errorMessage = error.error;
      } else if (error?.response?.data?.error) {
        errorMessage = error.response.data.error;
      } else if (error?.message) {
        errorMessage = error.message;
      }
      
      setError(errorMessage);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  // Fonction d'inscription avec meilleure gestion d'erreur
  const register = async (userData: { first_name: string; last_name: string; email: string; password: string; password2: string }) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('[Auth] Tentative d\'inscription avec:', { 
        email: userData.email, 
        first_name: userData.first_name,
        last_name: userData.last_name,
        password: '****'
      });
      
      const data = await authService.register({
        first_name: userData.first_name,
        last_name: userData.last_name,
        email: userData.email,
        password: userData.password,
        password2: userData.password2
      });
      console.log('[Auth] Inscription réussie:', data);
      
      if (data && data.user) {
        setUser(data.user);
        
        // Ajouter un délai avant la redirection
        setTimeout(() => {
          console.log('[Auth] Redirection vers les workflows');
          router.push('/workflows');
        }, 300);
      } else {
        throw new Error('Réponse du serveur invalide');
      }
    } catch (error: any) {
      console.error('[Auth] Erreur d\'inscription:', error);
      
      // Extraction optimisée du message d'erreur
      let errorMessage = 'Une erreur est survenue lors de l\'inscription';
      
      if (error?.error) {
        errorMessage = error.error;
      } else if (error?.response?.data?.error) {
        errorMessage = error.response.data.error;
      } else if (error?.response?.data?.username) {
        errorMessage = `Nom d'utilisateur: ${error.response.data.username}`;
      } else if (error?.response?.data?.email) {
        errorMessage = `Email: ${error.response.data.email}`;
      } else if (error?.response?.data?.password) {
        errorMessage = `Mot de passe: ${error.response.data.password}`;
      } else if (error?.message) {
        errorMessage = error.message;
      }
      
      setError(errorMessage);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  // Fonction de déconnexion
  const logout = async () => {
    setLoading(true);
    try {
      console.log('[Auth] Tentative de déconnexion');
      await authService.logout();
      setUser(null);
      
      // Redirection avec délai
      setTimeout(() => {
        console.log('[Auth] Redirection vers la page de connexion');
        router.push('/login');
      }, 100);
    } catch (error) {
      console.error('[Auth] Erreur de déconnexion:', error);
      // En cas d'erreur, nettoyer quand même
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        error,
        login,
        register,
        logout,
        isAuthenticated: !!user,
        clearError,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth doit être utilisé dans un AuthProvider');
  }
  return context;
}