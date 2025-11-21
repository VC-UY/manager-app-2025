# 🎯 MANAGER APP - INTERFACE DE GESTION DES WORKFLOWS

---

## 📝 DESCRIPTION DU PROJET

Le **Manager App** est l'application dédiée aux **gestionnaires de workflows** dans le système de calcul distribué volontaire. Il permet de créer, gérer et superviser les workflows de calcul, d'assigner des tâches aux volontaires et de suivre l'exécution en temps réel.

**Architecture** : Manager ↔ **Coordinator** ↔ Volontaire

**Rôle** : Interface de gestion des workflows, création de tâches, supervision des exécutions

---

## 🎯 OBJECTIFS

- **Objectif principal** : Fournir une interface complète pour la gestion des workflows de calcul distribué
- **Problématique** : Besoin d'un outil simple pour créer et gérer des workflows complexes sans expertise technique
- **Solution** : Application Django + Next.js avec authentification, gestion des tâches et communication Redis

---

## 🧩 FONCTIONNALITÉS

### 🔐 AUTHENTIFICATION UTILISATEUR
- **Inscription/Connexion** sécurisée avec tokens
- **Gestion des profils** utilisateur
- **Sessions persistantes** avec Next-Auth
- **Protection des routes** sensibles

### 📊 GESTION DES WORKFLOWS
- **Création de workflows** avec interface intuitive
- **Configuration des tâches** et sous-tâches
- **Estimation automatique** des ressources nécessaires
- **Templates** de workflows prédéfinis

### 👥 SUPERVISION DES VOLONTAIRES
- **Vue d'ensemble** des volontaires connectés
- **Monitoring** des performances et disponibilité
- **Attribution intelligente** des tâches
- **Historique** des exécutions

### 📈 TABLEAUX DE BORD
- **Dashboards interactifs** avec graphiques
- **Métriques en temps réel** des performances
- **Rapports détaillés** d'exécution
- **Alertes** et notifications

### 🔄 COMMUNICATION TEMPS RÉEL
- **WebSockets** pour les mises à jour live
- **Intégration Redis** pour la communication
- **Notifications push** d'état des tâches
- **Chat** avec les volontaires (si implémenté)

---

## 🚀 PRÉREQUIS SYSTÈME

### 🔧 **Logiciels Requis**
- **Python 3.8+** - [Télécharger Python](https://www.python.org/downloads/)
- **Node.js 18+** - [Télécharger Node.js](https://nodejs.org/)

- **Git** - [Installer Git](https://git-scm.com/downloads)

### 🌐 **Configuration Réseau**
- **Port 8001** : Backend Django (API REST + WebSockets)
- **Port 3000** : Frontend Next.js (développement)
- **Coordinateur** : Connexion au Coordinator App sur `COORDINATOR_IP:REDIS_PORT OR COORDINATOR_PROXY_PORT`

---

## 📦 INSTALLATION COMPLÈTE

### 1️⃣ **Cloner le Projet**
```bash
git clone https://github.com/VC-UY/manager-app-2025.git
```

### 2️⃣ **Installation Backend (Django)**
```bash
cd manager_backend

# Créer l'environnement virtuel
python -m venv exp-env

# Activer l'environnement virtuel
# Sur Linux/Mac :
source exp-env/bin/activate
# Sur Windows :
exp-env\Scripts\activate

# Installer les dépendances Python
pip install -r requirements.txt
```

### 3️⃣ **Installation Frontend (Next.js)**
```bash
cd ../manager_frontend

# Installer les dépendances Node.js
npm install
```

### 4️⃣ **Configuration Base de Données**
```bash
cd ../manager_backend

# Activer l'environnement virtuel
source exp-env/bin/activate

# Appliquer les migrations
python manage.py makemigrations
python manage.py migrate

# Créer un superutilisateur (optionnel)
python manage.py createsuperuser


daphne -b 0.0.0.0 -p 8001 websocket_service.asgi:application

daphne manager_backend.asgi:application -p 8003 -b 0.0.0.0

```


---

## ▶️ LANCEMENT DE L'APPLICATION

### 🚀 **Backend (3 terminaux requis)**

#### **Terminal 1 : Redis Server**
```bash
# Démarrer Redis sur le port par défaut
redis-server
```

#### **Terminal 2 : Coordinator App** (REQUIS)
```bash
# Se rendre dans le projet Coordinator
cd /path/to/Coordinator-App/coordinator_project

# Activer l'environnement du coordinator
source coordinator-env/bin/activate

# Démarrer le proxy Redis du coordinateur
python manage.py start_redis_proxy --redis-host localhost --redis-port 6379 --proxy-port 6380

# Dans un autre terminal, démarrer le backend coordinateur
daphne coordinator_project.asgi:application -p 8001 -b 0.0.0.0
```

#### **Terminal 3 : Backend Manager (Django/ASGI)**
```bash
cd manager_backend
source exp-env/bin/activate

# Lancer avec Daphne (ASGI pour WebSockets)
daphne -b 0.0.0.0 -p 8001 websocket_service.asgi:application
```

### 🖥️ **Frontend**

#### **Terminal 4 : Frontend Next.js**
```bash
cd manager_frontend

# Lancer le serveur de développement
npm run dev
```

### 🌐 **Accéder aux Applications**
- **Frontend Manager** : `http://localhost:3000`
- **Backend API** : `http://localhost:8001/api/`
- **Admin Django** : `http://localhost:8001/admin/`
- **Coordinator** : `http://localhost:5173` (doit être démarré)

---

## ⚙️ CONFIGURATION

### 🔧 **Configuration Backend**
Modifier `manager_backend/manager_backend/settings.py` :

```python
# Base de données SQLite (par défaut)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Redis Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# Coordinator Configuration
COORDINATOR_HOST = 'localhost'
COORDINATOR_PORT = 6380  # Port du proxy Redis du coordinateur

# CORS pour le frontend
CORS_ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'http://localhost:3001',
]

# Authentification
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
}
```

### 🌐 **Configuration Frontend**
Créer `manager_frontend/lib/config.ts` :

```typescript
export const API_BASE_URL = 'http://localhost:8001';
export const COORDINATOR_URL = 'ws://localhost:8001/ws';

// Configuration Next-Auth
export const NEXTAUTH_URL = 'http://localhost:3000';
export const NEXTAUTH_SECRET = 'your-secret-key-here';
```

### 🔒 **Variables d'Environnement**
Créer `manager_frontend/.env.local` :

```bash
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here
API_BASE_URL=http://localhost:8001
```

---

## 📁 STRUCTURE DU PROJET

```
ManagerApp/v2/
├── manager_backend/             # Backend Django
│   ├── manage.py               # Point d'entrée Django
│   ├── requirements.txt        # Dépendances Python
│   ├── db.sqlite3             # Base de données SQLite
│   ├── manager_backend/        # Configuration Django
│   │   ├── settings.py        # Configuration principale
│   │   ├── urls.py            # Routes principales
│   │   ├── wsgi.py            # Configuration WSGI
│   │   └── asgi.py            # Configuration ASGI
│   ├── workflows/             # App gestion workflows
│   │   ├── models.py          # Modèles Django
│   │   ├── views.py           # API REST workflows
│   │   ├── serializers.py     # Sérialiseurs DRF
│   │   ├── urls.py            # Routes workflows
│   │   └── auth.py            # Authentification
│   ├── tasks/                 # App gestion tâches
│   │   ├── models.py          # Modèles tâches
│   │   ├── views.py           # API REST tâches
│   │   └── serializers.py     # Sérialiseurs tâches
│   ├── volunteers/            # App gestion volontaires
│   │   ├── models.py          # Modèles volontaires
│   │   ├── views.py           # API REST volontaires
│   │   └── serializers.py     # Sérialiseurs volontaires
│   ├── websocket_service/     # Service WebSockets
│   │   ├── asgi.py            # Configuration ASGI
│   │   ├── routing.py         # Routes WebSocket
│   │   └── middleware.py      # Middleware auth
│   └── redis_communication/   # Communication Redis
└── manager_frontend/          # Frontend Next.js
    ├── package.json           # Dépendances Node.js
    ├── next.config.ts         # Configuration Next.js
    ├── tailwind.config.js     # Configuration Tailwind
    ├── app/                   # App Router Next.js 13+
    │   ├── layout.tsx         # Layout principal
    │   ├── page.tsx           # Page d'accueil
    │   ├── auth/              # Pages authentification
    │   ├── dashboard/         # Dashboard principal
    │   ├── workflows/         # Gestion workflows
    │   └── volunteers/        # Gestion volontaires
    ├── components/            # Composants React
    │   ├── ui/                # Composants UI réutilisables
    │   ├── forms/             # Formulaires
    │   └── charts/            # Graphiques
    ├── lib/                   # Utilitaires
    │   ├── api.ts             # Client API
    │   └── auth.ts            # Configuration auth
    └── public/                # Assets statiques
```

---

## 🔄 FONCTIONNEMENT

### 1. **Démarrage du Manager**
- Redis démarre pour la communication
- Le Coordinator App doit être en cours d'exécution
- Le backend Django/ASGI démarre avec WebSockets
- Le frontend Next.js se connecte au backend

### 2. **Authentification Manager**
- Manager s'inscrit/se connecte via l'interface Next.js
- Token d'authentification stocké côté client
- Communication sécurisée avec le Coordinator

### 3. **Création de Workflows**
- Interface intuitive pour définir les workflows
- Configuration des tâches et sous-tâches
- Estimation automatique des ressources
- Validation avant soumission

### 4. **Attribution des Tâches**
- Le workflow est soumis au Coordinator
- Le Coordinator distribue aux volontaires disponibles
- Suivi en temps réel via WebSockets
- Notifications de progression

### 5. **Supervision**
- Dashboard temps réel des exécutions
- Métriques de performance
- Gestion des erreurs et reprises
- Rapports détaillés

---

## 🛑 ARRÊTER L'APPLICATION

```bash
# Dans chaque terminal, appuyez sur :
Ctrl + C

# Arrêter Redis
redis-cli shutdown

# Désactiver l'environnement virtuel
deactivate
```

---

## 🐛 DÉPANNAGE

### ❌ **Erreur de connexion au Coordinator**
```bash
# Vérifier que le Coordinator est démarré
curl http://localhost:8001/api/system-health/

# Vérifier le proxy Redis du coordinateur
telnet localhost 6380
```

### ❌ **Erreur de connexion Redis**
```bash
# Vérifier que Redis fonctionne
redis-cli ping

# Redémarrer Redis si nécessaire
sudo systemctl restart redis-server
```

### ❌ **Erreur de migration Django**
```bash
cd manager_backend
source exp-env/bin/activate

# Réinitialiser les migrations si nécessaire
python manage.py migrate --run-syncdb
```

### ❌ **Erreur de build Next.js**
```bash
cd manager_frontend

# Nettoyer le cache
rm -rf .next node_modules
npm install
npm run dev
```

### ❌ **Problème d'authentification**
Vérifier les variables d'environnement :
```bash
# manager_frontend/.env.local
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here
```

---

## 📊 MONITORING

### 📈 **Tableau de Bord**
- **URL** : `http://localhost:3000/dashboard`
- **Fonctionnalités** :
  - Vue d'ensemble des workflows
  - Status des volontaires
  - Performances en temps réel
  - Historique des exécutions

### 📋 **API de Surveillance**
```bash
# Status des workflows
curl http://localhost:8001/api/workflows/

# Volontaires disponibles
curl http://localhost:8001/api/volunteers/

# Tâches en cours
curl http://localhost:8001/api/tasks/
```

### 🔍 **Logs et Debug**
```bash
# Logs du backend Django
tail -f manager_backend/server.log

# Logs Next.js (dans le terminal)
# Les erreurs s'affichent directement

# Debug API avec headers
curl -H "Authorization: Token YOUR_TOKEN" http://localhost:8001/api/workflows/
```

---

## 🔧 COMMANDES UTILES

### 📊 **Gestion Django**
```bash
cd manager_backend
source exp-env/bin/activate

# Créer des migrations
python manage.py makemigrations

# Appliquer les migrations
python manage.py migrate

# Shell Django
python manage.py shell

# Créer un superutilisateur
python manage.py createsuperuser
```

### 📡 **Gestion Next.js**
```bash
cd manager_frontend

# Build de production
npm run build

# Démarrer en production
npm start

# Linter le code
npm run lint
```

### 🐛 **Debug**
```bash
# Test de l'API backend
curl -X POST http://localhost:8001/api/workflows/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass"}'

# Test WebSocket
# Utiliser un client WebSocket pour tester ws://localhost:8001/ws/
```

---

## 🚀 PRODUCTION

### 🔧 **Déploiement Backend**
```bash
# Variables d'environnement production
export DJANGO_SETTINGS_MODULE=manager_backend.settings_prod
export DEBUG=False

# Collecter les fichiers statiques
python manage.py collectstatic

# Utiliser Gunicorn avec Workers ASGI
gunicorn manager_backend.asgi:application -k uvicorn.workers.UvicornWorker
```

### 🌐 **Déploiement Frontend**
```bash
# Build de production
npm run build

# Variables d'environnement
export NODE_ENV=production
export NEXTAUTH_URL=https://yourdomain.com

# Démarrer
npm start
```

---

## 📄 LICENCE

Ce projet est **open source** sous licence MIT.  
Réutilisation, modification et contribution autorisées.

---

## 👥 CONTRIBUTEURS

- **Équipe Manager** - Développement de l'interface de gestion
- **Équipe Coordinator** - Intégration système
- **Équipe Frontend** - Interface utilisateur Next.js

---

## 📞 SUPPORT

En cas de problème :

1. **Vérifier les prérequis** : Python, Node.js, Redis
2. **S'assurer que le Coordinator fonctionne** : Port 8001 et 6380
3. **Consulter les logs** : `server.log`, terminal Next.js
4. **Tester les connexions** : Redis (6379), Backend (8001), Frontend (3000)
5. **Contacter l'équipe via [mail](mailto:sergenoah91@gmail.com)** :  si le problème persiste

### 🔗 **Liens Utiles**
- Documentation Django : https://docs.djangoproject.com/
- Documentation Next.js : https://nextjs.org/docs
- Documentation React : https://reactjs.org/docs/
- Documentation Tailwind CSS : https://tailwindcss.com/docs

---

## 🛠️ CRÉATION DE WORKFLOWS PERSONNALISÉS

### 📋 **Vue d'ensemble**

L'application Manager permet d'ajouter facilement de nouveaux types de workflows personnalisés. Actuellement, deux types sont implémentés comme exemples :
- **ML_TRAINING** : Entraînement de modèles de machine learning (CIFAR-100)
- **OPEN_MALARIA** : Simulations épidémiologiques OpenMalaria

**⚠️ Note importante** : Les dépendances entre tâches sont prises en compte dans les modèles mais n'ont pas encore été testées dans l'assignation des tâches.

### 🔧 **Étapes pour créer un workflow personnalisé**

#### **1. Ajouter le type de workflow**

Modifier `manager_backend/workflows/models.py` :

```python
class WorkflowType(models.TextChoices):
    MATRIX_ADDITION = 'MATRIX_ADDITION', 'Addition de matrices de grande taille'
    MATRIX_MULTIPLICATION = 'MATRIX_MULTIPLICATION', 'Multiplication de matrices de grande taille'
    ML_TRAINING = 'ML_TRAINING', 'Entraînement de modèle machine learning'
    OPEN_MALARIA = 'OPEN_MALARIA', 'Simulation OpenMalaria'
    # ➕ Ajouter votre nouveau type ici
    YOUR_CUSTOM_TYPE = 'YOUR_CUSTOM_TYPE', 'Description de votre workflow'
```

#### **2. Créer la fonction de découpage**

Ajouter votre fonction dans `manager_backend/workflows/split_workflow.py` :

```python
def split_your_custom_workflow(workflow_instance: Workflow, logger: logging.Logger, **kwargs):
    """
    Découpe votre workflow personnalisé en tâches plus petites.
    
    Args:
        workflow_instance (Workflow): Instance du workflow à découper
        logger (logging.Logger): Logger pour les messages
        **kwargs: Paramètres spécifiques à votre workflow
    
    Returns:
        list: Liste des tâches créées
    """
    input_dir = os.path.join(workflow_instance.executable_path, "inputs")
    min_resources = get_min_volunteer_resources()
    
    # Configuration de l'image Docker pour vos tâches
    docker_img = {
        "name": "your-custom-image",
        "tag": "latest"
    }
    
    tasks = []
    
    # Exemple : créer plusieurs tâches basées sur vos données
    for i in range(kwargs.get('num_tasks', 4)):
        # Préparer les données d'entrée pour cette tâche
        # Exemple : générer des fichiers de configuration, découper des datasets, etc.
        
        # Calculer la taille des données d'entrée
        input_size = 100  # Remplacer par le calcul réel
        
        # Créer la tâche
        task = Task.objects.create(
            workflow=workflow_instance,
            name=f"Custom Task {i}",
            description=f"Description de la tâche {i}",
            command="your_command_here",  # Commande à exécuter dans le conteneur
            parameters=[],  # Paramètres additionnels si nécessaire
            input_files=[f"task_{i}/input.dat"],  # Fichiers d'entrée
            output_files=[f"task_{i}/output.dat"],  # Fichiers de sortie attendus
            status=TaskStatus.CREATED,
            parent_task=None,  # Pour les dépendances entre tâches
            is_subtask=False,
            progress=0,
            start_time=None,
            docker_info=docker_img,
            required_resources={
                "cpu": min_resources["min_cpu"],
                "ram": min_resources["min_ram"],
                "disk": min_resources["disk"],
            },
            estimated_max_time=300,  # Temps estimé en secondes
        )
        task.input_size = input_size
        task.save()
        tasks.append(task)
        logger.warning(f"Tâche {i}: {task} créée avec succès")
    
    workflow_instance.tasks.add(*tasks)
    workflow_instance.save()
    return tasks
```

#### **3. Intégrer dans la fonction principale**

Modifier la fonction `split_workflow` dans le même fichier :

```python
def split_workflow(id: uuid.UUID, workflow_type: WorkflowType, logger, **kwargs):
    """
    Découpe un workflow en tâches plus petites selon le type de workflow.
    """
    workflow_instance = Workflow.objects.get(id=id)
    
    if workflow_type == WorkflowType.ML_TRAINING:
        tasks = split_ml_training_workflow(workflow_instance, logger)
    elif workflow_type == WorkflowType.OPEN_MALARIA:
        num_tasks = kwargs.get('num_tasks')
        population_per_task = kwargs.get('population_per_task')
        if num_tasks is None or population_per_task is None:
            raise ValueError("num_tasks et population_per_task doivent être spécifiés pour OpenMalaria")
        tasks = split_openmalaria_workflow(workflow_instance, num_tasks, population_per_task, logger)
    # ➕ Ajouter votre workflow ici
    elif workflow_type == WorkflowType.YOUR_CUSTOM_TYPE:
        tasks = split_your_custom_workflow(workflow_instance, logger, **kwargs)
    else:
        raise ValueError(f"Type de workflow non supporté: {workflow_type}")
    
    return tasks
```

#### **4. Créer la vue de soumission**

Créer `manager_backend/workflows/your_custom_views.py` :

```python
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
import logging
import threading
import traceback
from workflows.models import Workflow, WorkflowStatus, WorkflowType
from websocket_service.client import notify_event

logger = logging.getLogger(__name__)

@api_view(['POST'])
def submit_your_custom_workflow_view(request, workflow_id):
    """
    Soumet un workflow personnalisé pour traitement.
    
    Args:
        workflow_id (str): ID du workflow
        request.data: Paramètres spécifiques à votre workflow
    
    Returns:
        JsonResponse: Statut de la soumission
    """
    try:
        # Récupérer et valider les paramètres
        custom_param1 = request.data.get('custom_param1')
        custom_param2 = request.data.get('custom_param2')
        
        # Validation des paramètres
        if not custom_param1:
            return JsonResponse({
                'error': 'custom_param1 est requis'
            }, status=400)
        
        # Récupérer le workflow
        workflow = get_object_or_404(Workflow, id=workflow_id)
        if workflow.workflow_type != WorkflowType.YOUR_CUSTOM_TYPE:
            return JsonResponse({
                'error': 'Le workflow doit être de type YOUR_CUSTOM_TYPE'
            }, status=400)
        
        # Notifier le début
        notify_event('workflow_status_change', {
            'workflow_id': str(workflow_id),
            'status': 'SUBMITTING',
            'message': 'Soumission du workflow personnalisé en cours...'
        })
        
        # Vérifier les volontaires disponibles
        from workflows.handlers import submit_workflow_handler
        success, response = submit_workflow_handler(str(workflow_id))
        
        if not success:
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow_id),
                'status': 'SUBMISSION_FAILED',
                'message': f"Échec de la soumission: {response.get('message', 'Erreur inconnue')}"
            })
            return JsonResponse({'success': False, 'response': response}, status=400)
        
        # Mettre à jour le statut
        workflow.status = WorkflowStatus.SPLITTING
        workflow.submitted_at = timezone.now()
        workflow.save()
        
        # Lancer le traitement asynchrone
        def process_workflow_async():
            thread_logger = logging.getLogger(f"workflow_thread_{workflow_id}")
            
            try:
                # Démarrer le serveur de fichiers
                from tasks.file_server import start_file_server
                server_port = start_file_server(workflow)
                
                # Découper le workflow
                from workflows.split_workflow import split_workflow
                tasks = split_workflow(
                    workflow.id, 
                    WorkflowType.YOUR_CUSTOM_TYPE, 
                    thread_logger,
                    custom_param1=custom_param1,
                    custom_param2=custom_param2
                )
                
                thread_logger.info(f"Découpage terminé, {len(tasks)} tâches créées")
                
                # Attribution des tâches (logique similaire à openmalaria_views.py)
                if response.get('volunteers'):
                    from tasks.scheduller import assign_workflow_to_volunteers
                    assignment_result = assign_workflow_to_volunteers(workflow, response.get('volunteers'))
                    
                    # Publier les assignations aux volontaires
                    # ... (logique de publication Redis)
                    
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': workflow.status,
                    'message': 'Processus de soumission terminé'
                })
                
            except Exception as e:
                thread_logger.error(f"Erreur lors du traitement: {e}")
                workflow.status = WorkflowStatus.FAILED
                workflow.save()
        
        # Démarrer le thread
        thread = threading.Thread(target=process_workflow_async)
        thread.daemon = True
        thread.start()
        
        return JsonResponse({
            'success': True, 
            'message': 'Workflow personnalisé soumis, traitement en cours'
        }, status=200)
        
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        return JsonResponse({'error': f'Erreur inattendue: {str(e)}'}, status=500)
```

#### **5. Ajouter les routes**

Modifier `manager_backend/workflows/urls.py` :

```python
from django.urls import path
from . import views, openmalaria_views, your_custom_views

urlpatterns = [
    # ... routes existantes ...
    
    # ➕ Ajouter votre route
    path('submit_your_custom/<uuid:workflow_id>/', 
         your_custom_views.submit_your_custom_workflow_view, 
         name='submit_your_custom_workflow'),
]
```

#### **6. Ajouter l'estimation des ressources**

Modifier `manager_backend/workflows/handlers.py` :

```python
def submit_workflow_handler(workflow_id: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None, timeout: int = 60):
    # ... code existant ...
    
    # Estimer les ressources
    estimated_resources = None
    if workflow.workflow_type == WorkflowType.ML_TRAINING:
        estimated_resources = estimate_ml_training_resources(workflow.input_data_size)
    elif workflow.workflow_type == WorkflowType.OPEN_MALARIA:
        estimated_resources = estimate_open_malaria_resources(workflow.metadata.get('num_task', 4))
    # ➕ Ajouter votre estimation
    elif workflow.workflow_type == WorkflowType.YOUR_CUSTOM_TYPE:
        estimated_resources = estimate_your_custom_resources(workflow.metadata)
    # ... reste du code ...
```

#### **7. Mettre à jour le frontend**

Ajouter votre type dans `manager_frontend/app/workflows/create/page.tsx` :

```typescript
const workflowTypes = [
  { value: 'MATRIX_ADDITION', label: 'Addition de matrices' },
  { value: 'MATRIX_MULTIPLICATION', label: 'Multiplication de matrices' },
  { value: 'ML_TRAINING', label: 'Entraînement ML' },
  { value: 'OPEN_MALARIA', label: 'Simulation OpenMalaria' },
  // ➕ Ajouter votre type
  { value: 'YOUR_CUSTOM_TYPE', label: 'Votre workflow personnalisé' },
];
```

### 🐳 **Préparation du conteneur Docker**

Votre workflow aura besoin d'une image Docker qui devra être disponible chez chacun des volontaires ou sur le docker hub. Créer un `Dockerfile` :

```dockerfile
FROM python:3.9-slim

# Installer les dépendances de votre application
RUN apt-get update && apt-get install -y \
    your-dependencies \
    && rm -rf /var/lib/apt/lists/*

# Copier votre code
COPY your_app/ /app/
WORKDIR /app

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Point d'entrée
ENTRYPOINT ["python", "your_main_script.py"]
```

### 📁 **Structure des fichiers**

Organiser vos fichiers comme suit :

```
workflows/examples/your_custom/
├── inputs/                 # Dossier pour les données d'entrée générées
│   ├── task_0/
│   │   └── input.dat
│   ├── task_1/
│   │   └── input.dat
│   └── ...
├── outputs/               # Dossier pour les résultats
│   ├── task_0/
│   │   └── output.dat
│   └── ...
├── scripts/               # Scripts utilitaires
│   ├── prepare_data.py
│   └── process_results.py
└── docker/               # Configuration Docker
    ├── Dockerfile
    └── requirements.txt
```

### 🔗 **Gestion des dépendances entre tâches**

Pour créer des tâches avec dépendances :

```python
# Créer la tâche parent
parent_task = Task.objects.create(
    workflow=workflow_instance,
    name="Tâche initiale",
    # ... autres paramètres ...
)

# Créer une tâche dépendante
dependent_task = Task.objects.create(
    workflow=workflow_instance,
    name="Tâche dépendante",
    parent_task=parent_task,  # ← Définir la dépendance
    is_subtask=True,
    # ... autres paramètres ...
)
```

### 📊 **Estimation des ressources**

Implémenter une fonction d'estimation :

```python
def estimate_your_custom_resources(metadata):
    """
    Estime les ressources nécessdjango.db.utils.OperationalError: no such table: workflows_useraires pour votre workflow.
    
    Args:
        metadata (dict): Métadonnées du workflow
    
    Returns:
        dict: Ressources estimées
    """
    base_cpu = 2
    base_ram = 1024  # MB
    base_disk = 2048  # MB
    
    # Adapter selon vos paramètres
    num_tasks = metadata.get('num_tasks', 4)
    data_size = metadata.get('data_size', 100)
    
    return {
        "cpu": base_cpu,
        "ram": base_ram + (data_size * 10),
        "disk": base_disk + (data_size * 20),
        "estimated_time": num_tasks * 300,  # secondes
    }
```

### 🧪 **Test de votre workflow**

1. **Créer un workflow de test** via l'interface
2. **Soumettre avec des paramètres simples**
3. **Vérifier les logs** dans les terminaux
4. **Valider la génération des tâches**
5. **Tester l'assignation** aux volontaires

### 📝 **Exemple complet : Workflow de traitement d'images**

Voici un exemple concret pour un workflow de traitement d'images :

```python
def split_image_processing_workflow(workflow_instance: Workflow, logger: logging.Logger, **kwargs):
    """Découpe un workflow de traitement d'images en tâches."""
    
    input_dir = os.path.join(workflow_instance.executable_path, "inputs")
    image_folder = kwargs.get('image_folder', '/path/to/images')
    filter_type = kwargs.get('filter_type', 'blur')
    
    # Lister les images à traiter
    import os
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Grouper les images par batch
    batch_size = kwargs.get('batch_size', 10)
    batches = [image_files[i:i+batch_size] 
               for i in range(0, len(image_files), batch_size)]
    
    tasks = []
    for i, batch in enumerate(batches):
        # Créer le dossier d'entrée pour ce batch
        batch_dir = os.path.join(input_dir, f"batch_{i}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Créer un fichier de configuration
        config = {
            'images': batch,
            'filter_type': filter_type,
            'output_format': 'png'
        }
        
        import json
        with open(os.path.join(batch_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        # Créer la tâche
        task = Task.objects.create(
            workflow=workflow_instance,
            name=f"Image Processing Batch {i}",
            description=f"Traitement de {len(batch)} images avec filtre {filter_type}",
            command=f"python process_images.py --config /input/config.json --output /output/",
            parameters=[],
            input_files=[f"batch_{i}/config.json"],
            output_files=[f"batch_{i}/processed/"],
            status=TaskStatus.CREATED,
            docker_info={
                "name": "image-processor",
                "tag": "latest"
            },
            required_resources={
                "cpu": 2,
                "ram": 2048,
                "disk": len(batch) * 50,  # 50MB par image
            },
            estimated_max_time=len(batch) * 30,  # 30s par image
        )
        
        tasks.append(task)
    
    workflow_instance.tasks.add(*tasks)
    workflow_instance.save()
    return tasks
```

Cette architecture modulaire permet d'ajouter facilement de nouveaux types de workflows tout en maintenant la compatibilité avec l'infrastructure existante.

---