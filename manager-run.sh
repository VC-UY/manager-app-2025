#!/bin/bash

# Script d'installation et de lancement automatique pour l'application Manager
# Ce script installe toutes les dépendances et lance l'application automatiquement

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
COORDINATOR_IP="173.249.38.251"  # IP du serveur coordinator déployé
COORDINATOR_PORT="80"répertoire
BACKEND_PORT="8002"
FRONTEND_PORT="3000"

echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}    Application Manager - Installation Automatique${NC}"
echo -e "${GREEN}======================================================${NC}\n"

# Obtenir le  du script (qui est maintenant dans manager-app-2025/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANAGER_DIR="$SCRIPT_DIR"
VENV_DIR="$SCRIPT_DIR/venv"

# Fonction pour afficher les erreurs
error_exit() {
    echo -e "\n${RED}ERREUR: $1${NC}" >&2
    exit 1
}

# Fonction pour vérifier si une commande existe
command_exists() {
    command -v "$1" &> /dev/null
}

# Fonction pour détecter le système d'exploitation
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "debian"
        elif [ -f /etc/redhat-release ]; then
            echo "redhat"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

OS_TYPE=$(detect_os)
echo -e "${BLUE}Système d'exploitation détecté: $OS_TYPE${NC}\n"

# Vérifier si le script est exécuté avec sudo si nécessaire
if [ "$EUID" -ne 0 ] && [ "$OS_TYPE" != "macos" ]; then
    echo -e "${YELLOW}Ce script nécessite des privilèges administrateur pour installer les dépendances.${NC}"
    echo -e "${YELLOW}Relancement avec sudo...${NC}\n"
    exec sudo bash "$0" "$@"
fi

# Fonction pour nettoyer les dépôts problématiques
cleanup_apt_sources() {
    echo -e "${YELLOW}Nettoyage des dépôts APT problématiques...${NC}"
    
    # Supprimer les doublons VSCode si présents
    if [ -f /etc/apt/sources.list.d/archive_uri-https_packages_microsoft_com_repos_code-noble.list ] && \
       [ -f /etc/apt/sources.list.d/vscode.sources ]; then
        echo -e "${YELLOW}  - Suppression des doublons VSCode${NC}"
        rm -f /etc/apt/sources.list.d/archive_uri-https_packages_microsoft_com_repos_code-noble.list
    fi
    
    # Gérer le dépôt R-project problématique
    if [ -f /etc/apt/sources.list.d/cran-r.list ] || \
       [ -f /etc/apt/sources.list.d/archive_uri-https_cloud_r-project_org_bin_linux_ubuntu-noble.list ]; then
        echo -e "${YELLOW}  - Correction du dépôt R-project${NC}"
        
        # Essayer d'ajouter la clé GPG manquante
        if apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9 2>/dev/null; then
            echo -e "${GREEN}    Clé GPG R-project ajoutée avec succès${NC}"
        else
            # Si l'ajout de la clé échoue, supprimer les dépôts R
            echo -e "${YELLOW}    Impossible d'ajouter la clé. Suppression des dépôts R${NC}"
            rm -f /etc/apt/sources.list.d/cran-r.list
            rm -f /etc/apt/sources.list.d/archive_uri-https_cloud_r-project_org_bin_linux_ubuntu-noble.list
        fi
    fi
}

# Fonction d'installation pour Debian/Ubuntu
install_debian() {
    # Nettoyer les sources problématiques d'abord
    cleanup_apt_sources
    
    echo -e "${YELLOW}[1/5] Mise à jour des paquets système...${NC}"
    # Utiliser --allow-releaseinfo-change pour Ubuntu et ignorer les erreurs non critiques
    apt-get update --allow-releaseinfo-change 2>&1 | grep -v "^W:" || {
        echo -e "${YELLOW}Avertissement: Certains dépôts ont généré des erreurs, mais on continue...${NC}"
    }

    echo -e "${YELLOW}[2/6] Installation de Python et pip...${NC}"
    apt-get install -y python3 python3-pip python3-venv python3-dev build-essential

    echo -e "${YELLOW}[3/5] Runtime vc-uyr (bundles) — pas de Docker pour l'exécution des tâches${NC}"
    echo -e "${GREEN}Les workflows volontaires passent par des bundles vc-uyr (voir volontaire/run.sh).${NC}"

    echo -e "${YELLOW}[4/5] Installation de Redis...${NC}"
    if ! command_exists redis-server; then
        apt-get install -y redis-server
        systemctl enable redis-server
        systemctl start redis-server
    else
        echo -e "${GREEN}Redis est déjà installé${NC}"
    fi

    echo -e "${YELLOW}[5/5] Installation de Git et autres outils...${NC}"
    apt-get install -y git curl wget tmux

    echo -e "${YELLOW}Installation de Node.js et npm...${NC}"
    if ! command_exists node; then
        # Utiliser Node.js 20.x (LTS) au lieu de 18.x (obsolète)
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
        apt-get install -y nodejs
    else
        NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
        echo -e "${GREEN}Node.js est déjà installé (version $(node --version))${NC}"
        
        # Vérifier si la version est trop ancienne (< 18)
        if [ "$NODE_VERSION" -lt 18 ]; then
            echo -e "${YELLOW}Version de Node.js trop ancienne. Mise à jour vers Node.js 20.x LTS...${NC}"
            curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
            apt-get install -y nodejs
        fi
    fi
    
    # Recharger les variables d'environnement et le PATH
    hash -r 2>/dev/null || true
    export PATH="/usr/bin:/usr/local/bin:$PATH"
    
    # Vérifier que npm est bien installé avec plusieurs tentatives
    if ! command_exists npm; then
        echo -e "${YELLOW}npm non trouvé dans le PATH, tentative d'installation séparée...${NC}"
        apt-get install -y npm || true
        hash -r 2>/dev/null || true
    fi
    
    # Dernière vérification
    if command_exists npm; then
        echo -e "${GREEN}Node.js $(node --version) et npm $(npm --version) installés${NC}"
    else
        echo -e "${RED}ATTENTION: npm n'a pas été installé correctement.${NC}"
        echo -e "${YELLOW}Installation de npm via alternative method...${NC}"
        # Installer npm via le package Ubuntu si nécessaire
        apt-get install -y npm
        hash -r 2>/dev/null || true
        
        if ! command_exists npm; then
            error_exit "npm n'a pas pu être installé. Utilisez: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash && nvm install 20"
        fi
    fi
}

# Fonction d'installation pour RedHat/CentOS/Fedora
install_redhat() {
    echo -e "${YELLOW}[1/5] Installation de Python et pip...${NC}"
    yum install -y python3 python3-pip python3-devel gcc

    echo -e "${YELLOW}[2/5] Runtime vc-uyr — pas de Docker pour les tâches volontaires${NC}"

    echo -e "${YELLOW}[3/5] Installation de Redis...${NC}"
    if ! command_exists redis-server; then
        yum install -y redis
        systemctl enable redis
        systemctl start redis
    else
        echo -e "${GREEN}Redis est déjà installé${NC}"
    fi

    echo -e "${YELLOW}[4/5] Installation de Git et autres outils...${NC}"
    yum install -y git curl wget tmux

    echo -e "${YELLOW}[5/5] Installation de Node.js et npm...${NC}"
    if ! command_exists node; then
        curl -fsSL https://rpm.nodesource.com/setup_18.x | bash -
        yum install -y nodejs
    else
        echo -e "${GREEN}Node.js est déjà installé${NC}"
    fi
}

# Fonction d'installation pour macOS
install_macos() {
    if ! command_exists brew; then
        error_exit "Homebrew n'est pas installé. Installez-le depuis https://brew.sh"
    fi

    echo -e "${YELLOW}[1/4] Installation de Python...${NC}"
    brew install python3

    echo -e "${YELLOW}[2/4] Installation de Redis...${NC}"
    brew install redis
    brew services start redis

    echo -e "${YELLOW}[3/4] Installation de Git et autres outils...${NC}"
    brew install git tmux

    echo -e "${YELLOW}[4/4] Installation de Node.js...${NC}"
    brew install node
}

# Installer les dépendances système selon l'OS
echo -e "\n${GREEN}=== Installation des dépendances système ===${NC}\n"

case $OS_TYPE in
    debian)
        install_debian
        ;;
    redhat)
        install_redhat
        ;;
    macos)
        install_macos
        ;;
    *)
        error_exit "Système d'exploitation non supporté: $OS_TYPE"
        ;;
esac

echo -e "\n${GREEN}=== Installation des dépendances Python ===${NC}\n"

# Créer l'environnement virtuel si nécessaire
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Création de l'environnement virtuel...${NC}"
    python3 -m venv "$VENV_DIR"
else
    echo -e "${GREEN}L'environnement virtuel existe déjà${NC}"
fi

# Activer l'environnement virtuel
source "$VENV_DIR/bin/activate"

# Vérifier que l'environnement virtuel est actif
if [ -z "$VIRTUAL_ENV" ]; then
    error_exit "Impossible d'activer l'environnement virtuel"
fi

echo -e "${GREEN}Environnement virtuel activé: $VIRTUAL_ENV${NC}"

# Installer les dépendances Python du manager
echo -e "${YELLOW}Installation des dépendances Python du backend...${NC}"
if [ ! -d "$MANAGER_DIR/manager_backend" ]; then
    error_exit "Le répertoire manager_backend n'existe pas"
fi

cd "$MANAGER_DIR/manager_backend"

if [ ! -f "requirements.txt" ]; then
    error_exit "Le fichier requirements.txt n'existe pas"
fi

pip install --upgrade pip setuptools wheel
pip install --upgrade -r requirements.txt

# Vérifier et installer PyTorch CPU-only si nécessaire
echo -e "${YELLOW}Vérification de PyTorch...${NC}"
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}PyTorch n'est pas installé. Installation de la version CPU-only...${NC}"

    # Corriger les permissions du venv si nécessaire
    if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
        echo -e "${YELLOW}Correction des permissions de l'environnement virtuel...${NC}"
        chown -R "$SUDO_USER:$SUDO_USER" "$VENV_DIR"
    fi

    # Installer PyTorch CPU-only (plus léger, ~200 MB au lieu de 2.3 GB)
    # Utiliser extra-index-url pour avoir un fallback sur PyPI
    echo -e "${YELLOW}Installation de PyTorch CPU-only depuis PyPI...${NC}"
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu || {
        echo -e "${YELLOW}Index CPU non accessible, installation depuis PyPI standard (version CPU)...${NC}"
        pip install torch torchvision torchaudio
    }
    echo -e "${GREEN}PyTorch installé avec succès${NC}"
else
    echo -e "${GREEN}PyTorch est déjà installé${NC}"
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}Version: $TORCH_VERSION${NC}"
fi

echo -e "\n${GREEN}=== Installation des dépendances Node.js ===${NC}\n"

# Installer les dépendances Node.js du frontend
echo -e "${YELLOW}Installation des dépendances Node.js du frontend...${NC}"
if [ ! -d "$MANAGER_DIR/manager_frontend" ]; then
    error_exit "Le répertoire manager_frontend n'existe pas"
fi

cd "$MANAGER_DIR/manager_frontend"

if [ ! -f "package.json" ]; then
    error_exit "Le fichier package.json n'existe pas"
fi

npm install

echo -e "\n${GREEN}=== Configuration de l'application ===${NC}\n"

# Mettre à jour le fichier de configuration backend avec l'IP du coordinator
echo -e "${YELLOW}Configuration de la connexion au coordinator...${NC}"

# Créer un fichier .env s'il n'existe pas
ENV_FILE="$MANAGER_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Création du fichier .env...${NC}"
    cat > "$ENV_FILE" << EOF
COORDINATOR_HOST=$COORDINATOR_IP
COORDINATOR_PROXY_PORT=$COORDINATOR_PORT
EOF
    echo -e "${GREEN}Fichier .env créé${NC}"
else
    echo -e "${GREEN}Fichier .env existe déjà${NC}"
fi

# Exporter les variables d'environnement
export COORDINATOR_HOST="$COORDINATOR_IP"
export COORDINATOR_PROXY_PORT="$COORDINATOR_PORT"

echo -e "${GREEN}Variables d'environnement configurées:${NC}"
echo -e "  COORDINATOR_HOST=$COORDINATOR_HOST"
echo -e "  COORDINATOR_PROXY_PORT=$COORDINATOR_PROXY_PORT"

# Appliquer les migrations
echo -e "${YELLOW}Application des migrations de base de données...${NC}"
cd "$MANAGER_DIR/manager_backend"
python manage.py migrate

# Créer un superutilisateur si nécessaire (optionnel)
echo -e "${YELLOW}Création d'un superutilisateur (optionnel)...${NC}"
echo -e "${BLUE}Laissez vide pour ignorer${NC}"
python manage.py createsuperuser --noinput --email admin@example.com 2>/dev/null || true

echo -e "\n${GREEN}=== Démarrage de l'application Manager ===${NC}\n"

# Vérifier si tmux est disponible pour lancer backend et frontend en parallèle
if command_exists tmux; then
    echo -e "${BLUE}Lancement du backend et du frontend dans des sessions tmux séparées${NC}\n"

    SESSION_NAME="manager-app"

    # Tuer la session si elle existe déjà
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

    # Créer une nouvelle session tmux avec le backend
    tmux new-session -d -s "$SESSION_NAME" -n "backend"
    tmux send-keys -t "$SESSION_NAME:0" "cd $MANAGER_DIR/manager_backend && source $VENV_DIR/bin/activate && daphne websocket_service.asgi:application -p $BACKEND_PORT -b 0.0.0.0" C-m

    # Créer une nouvelle fenêtre pour le frontend
    tmux new-window -t "$SESSION_NAME:1" -n "frontend"
    tmux send-keys -t "$SESSION_NAME:1" "cd $MANAGER_DIR/manager_frontend && npm run dev" C-m

    echo -e "${GREEN}Application Manager démarrée avec succès${NC}\n"
    echo -e "${BLUE}Backend: http://localhost:$BACKEND_PORT${NC}"
    echo -e "${BLUE}Frontend: http://localhost:$FRONTEND_PORT${NC}"
    echo -e "${BLUE}Coordinator: http://$COORDINATOR_IP${NC}\n"

    echo -e "${YELLOW}Ouvrez votre navigateur à l'adresse: http://localhost:$FRONTEND_PORT${NC}\n"

    echo -e "${YELLOW}Commandes utiles:${NC}"
    echo -e "  Attacher à la session: tmux attach-session -t $SESSION_NAME"
    echo -e "  Naviguer entre les fenêtres: Ctrl+b puis 0 (backend) ou 1 (frontend)"
    echo -e "  Détacher de la session: Ctrl+b puis d"
    echo -e "  Arrêter l'application: tmux kill-session -t $SESSION_NAME"
    echo -e "\n${GREEN}Attachement à la session tmux...${NC}"

    # Attacher à la session
    tmux attach-session -t "$SESSION_NAME"
else
    echo -e "${YELLOW}tmux n'est pas disponible. Lancement du backend uniquement.${NC}"
    echo -e "${YELLOW}Veuillez lancer le frontend manuellement dans un autre terminal:${NC}"
    echo -e "  cd $MANAGER_DIR/manager_frontend && npm run dev\n"

    echo -e "${BLUE}Backend: http://localhost:$BACKEND_PORT${NC}"
    echo -e "${BLUE}Coordinator: http://$COORDINATOR_IP${NC}\n"

    cd "$MANAGER_DIR/manager_backend"
    source "$VENV_DIR/bin/activate"
    daphne websocket_service.asgi:application -p "$BACKEND_PORT" -b 0.0.0.0
fi