# Application Manager - Guide d'Installation

Ce guide vous aidera à installer et lancer l'application Manager sur votre machine.

## Installation Automatique (Recommandée)

Une seule commande suffit pour tout installer et lancer l'application:

```bash
./manager-run.sh
```

Ce script effectue automatiquement:
1. Detection du système d'exploitation
2. Installation de toutes les dépendances nécessaires (Python, Node.js, Docker, Redis, etc.)
3. Creation de l'environnement virtuel Python
4. Installation des dépendances Python et Node.js
5. Configuration de la connexion au serveur central
6. Application des migrations de base de données
7. Lancement du backend et du frontend dans des sessions tmux

## Configuration Requise

### Systèmes d'exploitation supportés
- Ubuntu/Debian Linux
- CentOS/RedHat/Fedora
- macOS (avec Homebrew)

### Connexion Internet
Une connexion Internet est requise pour:
- Télécharger les dépendances
- Se connecter au serveur central (Coordinator)

## Utilisation

### Première Installation

```bash
# 1. Rendre le script exécutable (une seule fois)
chmod +x manager-run.sh

# 2. Lancer l'installation et l'application
./manager-run.sh
```

Le script vous demandera le mot de passe administrateur pour installer les dépendances système.

### Accès à l'Application

Une fois l'installation terminée, l'application sera accessible à:

**Frontend (Interface Utilisateur):**
```
http://localhost:3000
```

**Backend (API):**
```
http://localhost:8002
```

Ouvrez l'adresse du frontend dans votre navigateur web.

### Arrêt de l'Application

```bash
# Arrêter la session tmux
tmux kill-session -t manager-app
```

Ou appuyez sur `Ctrl+C` dans le terminal si tmux n'est pas utilisé.

### Relancer l'Application

Pour relancer l'application après l'avoir arrêtée:

```bash
./manager-run.sh
```

Le script détectera que les dépendances sont déjà installées et lancera directement l'application.

### Navigation dans tmux

Si l'application est lancée avec tmux, vous pouvez:

```bash
# Attacher à la session
tmux attach-session -t manager-app

# Naviguer entre les fenêtres
# Backend: Ctrl+b puis 0
# Frontend: Ctrl+b puis 1

# Détacher de la session (sans arrêter l'application)
# Ctrl+b puis d
```

## Configuration Avancée

### Modifier l'Adresse du Serveur Central

Si vous devez vous connecter à un serveur différent, modifiez le fichier `.env`:

```bash
# Editer le fichier .env
nano .env

# Modifier les valeurs suivantes:
COORDINATOR_HOST=173.249.38.251
COORDINATOR_PROXY_PORT=80
```

Ou directement dans le script `manager-run.sh`:

```bash
# Modifier ces lignes au début du script:
COORDINATOR_IP="173.249.38.251"
COORDINATOR_PORT="80"
```

### Ports Utilisés

L'application utilise les ports suivants:
- **3000**: Port du frontend (Next.js)
- **8002**: Port du backend (Django/Daphne)
- Les services externes (Redis, Docker) utilisent leurs ports par défaut

### Modifier les Ports

Pour utiliser des ports différents, modifiez le script `manager-run.sh`:

```bash
BACKEND_PORT="8002"    # Changer ce port si nécessaire
FRONTEND_PORT="3000"   # Changer ce port si nécessaire
```

## Dépannage

### Problème: "Permission denied"

Si vous obtenez cette erreur, rendez le script exécutable:

```bash
chmod +x manager-run.sh
```

### Problème: "Port déjà utilisé"

Si un port est déjà utilisé, trouvez et arrêtez le processus:

```bash
# Pour le port 3000 (frontend)
sudo lsof -ti:3000 | xargs kill -9

# Pour le port 8002 (backend)
sudo lsof -ti:8002 | xargs kill -9
```

### Problème: "Impossible de se connecter au serveur central"

Vérifiez:
1. Votre connexion Internet
2. Que le serveur central est bien en ligne (173.249.38.251)
3. Qu'aucun pare-feu ne bloque la connexion

```bash
# Tester la connexion au serveur
ping 173.249.38.251
curl http://173.249.38.251
```

### Problème: "npm: command not found"

Node.js n'est pas installé. Le script devrait l'installer automatiquement, mais vous pouvez aussi:

```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo bash -
sudo apt-get install -y nodejs

# macOS
brew install node
```

### Problème: "Docker ne démarre pas"

Pour les utilisateurs Linux, ajoutez votre utilisateur au groupe docker:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Puis relancez le script.

### Problème: Frontend ne charge pas

Si le frontend ne se lance pas:

```bash
# Aller dans le dossier frontend
cd manager_frontend

# Réinstaller les dépendances
rm -rf node_modules package-lock.json
npm install

# Lancer manuellement
npm run dev
```

### Réinstallation Complète

Si vous rencontrez des problèmes persistants:

```bash
# Supprimer l'environnement virtuel
rm -rf venv

# Supprimer les node_modules
rm -rf manager_frontend/node_modules

# Supprimer la base de données
rm -rf manager_backend/db.sqlite3

# Relancer l'installation
./manager-run.sh
```

## Création d'un Compte Utilisateur

Au premier lancement, vous devrez peut-être créer un compte superutilisateur:

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Aller dans le dossier backend
cd manager_backend

# Créer un superutilisateur
python manage.py createsuperuser

# Suivre les instructions
```

## Support

En cas de problème:
1. Consultez la section Dépannage ci-dessus
2. Vérifiez les messages d'erreur affichés par le script
3. Vérifiez les logs dans la session tmux
4. Contactez l'administrateur système

## Notes Importantes

- L'application se connecte automatiquement au serveur central déployé
- Toutes les dépendances sont installées automatiquement
- L'environnement virtuel Python est créé localement dans le dossier `venv/`
- Les modules Node.js sont installés dans `manager_frontend/node_modules/`
- La base de données locale est stockée dans `manager_backend/db.sqlite3`
- Le script détecte automatiquement votre système d'exploitation
- Le backend et le frontend sont lancés dans des sessions tmux séparées pour faciliter la gestion
