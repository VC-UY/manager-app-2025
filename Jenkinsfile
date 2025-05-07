pipeline {
    agent any
    
    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-credentials')
        IMAGE_NAME = "patricehub/workflow_manager"
        GITHUB_CREDENTIALS = credentials('github-token')
    }
    
    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/neussi/workflow_manager.git',
                    branch: 'main',
                    credentialsId: 'github-token'
            }
        }
        
        stage('Docker Pull Base Images') {
            steps {
                script {
                    sh 'docker pull python:3.9-slim'
                    sh 'docker pull node:18-alpine'
                }
            }
        }
        
        stage('Check Files') {
            steps {
                script {
                    sh 'ls -la'
                    sh 'ls -la backend/'
                    sh 'ls -la frontend/'
                    sh 'ls -la docker/'
                }
            }
        }
        
        stage('Update Dockerfiles') {
            steps {
                script {
                    // Update backend Dockerfile
                    sh '''
                    cat > docker/backend.Dockerfile <<'EOF'
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Mise à jour des paquets
RUN apt-get update || echo "apt-get update failed, continuing anyway"

# Copier requirements.txt
COPY backend/requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org \
    || echo "pip install failed, continuing anyway"

# Copier le code du backend
COPY backend/ .

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=centre_sante_dipita.settings \
    DEBUG=1

# Exposer le port Django
EXPOSE 8000

# Commande de lancement
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
EOF
                    '''
                    
                    // Update frontend Dockerfile
                    sh '''
                    cat > docker/frontend.Dockerfile <<'EOF'
FROM node:18-alpine

# Mise à jour des paquets
RUN apk update || echo "apk update failed, continuing anyway"
RUN apk add --no-cache bash || echo "apk add failed, continuing anyway"

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers package.json et package-lock.json
COPY package.json package-lock.json ./

# Installer les dépendances avec des options plus robustes
RUN npm config set fetch-retry-mintimeout 20000 \
    && npm config set fetch-retry-maxtimeout 120000 \
    && npm install --no-fund --no-audit --prefer-offline || echo "npm install failed, continuing anyway"

# Copier le reste des fichiers du frontend dans le conteneur
COPY frontend/ ./frontend/

# Exposer le port utilisé par l'application React
EXPOSE 3000

# Lancer l'application React
CMD ["npm", "start"]
EOF
                    '''
                }
            }
        }
        
        stage('Build Backend Image') {
            steps {
                script {
                    sh '''
                    docker build -t $IMAGE_NAME-backend:latest -f docker/backend.Dockerfile . || (
                        echo "First build attempt failed, retrying..." && 
                        sleep 10 && 
                        docker build -t $IMAGE_NAME-backend:latest -f docker/backend.Dockerfile .
                    )
                    '''
                }
            }
        }
        
        stage('Build Frontend Image') {
            steps {
                script {
                    sh '''
                    docker build -t $IMAGE_NAME-frontend:latest -f docker/frontend.Dockerfile . || (
                        echo "First build attempt failed, retrying..." && 
                        sleep 10 && 
                        docker build -t $IMAGE_NAME-frontend:latest -f docker/frontend.Dockerfile .
                    )
                    '''
                }
            }
        }
        
        stage('Login DockerHub') {
            steps {
                script {
                    sh "echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin"
                }
            }
        }
        
        stage('Push Images') {
            steps {
                script {
                    sh '''
                    docker push $IMAGE_NAME-backend:latest || (
                        echo "First push attempt failed, retrying..." && 
                        sleep 10 && 
                        docker push $IMAGE_NAME-backend:latest
                    )
                    
                    docker push $IMAGE_NAME-frontend:latest || (
                        echo "First push attempt failed, retrying..." && 
                        sleep 10 && 
                        docker push $IMAGE_NAME-frontend:latest
                    )
                    '''
                }
            }
        }
        
        stage('Logout from DockerHub') {
            steps {
                sh 'docker logout'
            }
        }
    }
}