# Manager de Workflow pour Système de Calcul Distribué Volontaire

## Vue d'ensemble

Le Manager de Workflow est un composant central d'un système de calcul distribué reposant sur la participation volontaire. Il sert d'interface principale pour la soumission, la gestion et le suivi des tâches de calcul dans des environnements à ressources limitées.

Ce projet répond à des défis spécifiques liés à l'accès restreint aux infrastructures de calcul haute performance, aux environnements contraints et hétérogènes, et aux risques de pannes et déconnexions réseau.

## Fonctionnalités principales

- Création et édition de workflows complexes
- Découpage automatique de tâches en sous-tâches pour optimiser le parallélisme
- Attribution intelligente des tâches aux volontaires disponibles
- Suivi en temps réel de l'exécution des workflows
- Agrégation fiable des résultats
- Reprise sur erreur en cas de défaillance
- Interface utilisateur intuitive et réactive

## Technologies utilisées

### Backend
- **Django** : Framework web Python pour l'API REST et la logique métier
- **Django REST Framework** : API REST complète
- **Django Channels** : Communication temps réel via WebSockets
- **MongoDB** : Base de données NoSQL pour le stockage flexible des workflows et tâches
- **PyMongo** : Connecteur MongoDB pour Python
- **Celery** : Traitement asynchrone des tâches

### Frontend
- **React** : Bibliothèque JavaScript pour l'interface utilisateur
- **Material-UI** : Composants React prédéfinis suivant le Material Design
- **Redux** : Gestion de l'état de l'application
- **React Router** : Navigation entre les différentes vues
- **Axios** : Client HTTP pour les requêtes API
- **Socket.io** : Communication bidirectionnelle temps réel
- **D3.js** : Visualisations de données
- **ReactFlow** : Visualisation et édition des workflows

### Communication
- **RESTful API** : Interface principale entre frontend et backend
- **WebSockets** : Notifications et mises à jour en temps réel
- **Pub/Sub** : Communication asynchrone avec le système de coordination

### DevOps
- **Docker** : Conteneurisation de l'application
- **Docker Compose** : Orchestration des services
- **GitHub Actions** : Intégration et déploiement continus

## Structure du projet

```
manager-workflow/
├── backend/                # Django backend
│   ├── core/               # Configuration Django principale
│   ├── workflows/          # Gestion des workflows
│   ├── tasks/              # Gestion des tâches
│   ├── communication/      # Communication avec coordinateur
│   ├── results/            # Gestion des résultats
│   └── users/              # Gestion des utilisateurs
├── frontend/               # React frontend
│   ├── public/             # Fichiers statiques
│   ├── src/
│   │   ├── components/     # Composants React
│   │   ├── pages/          # Pages de l'application
│   │   ├── store/          # État global Redux
│   │   ├── services/       # Services API
│   │   └── utils/          # Utilitaires
└── docker/                 # Configuration Docker
    ├── docker-compose.yml
    ├── backend.Dockerfile
    └── frontend.Dockerfile
```

## Modèles principaux MongoDB

### Collection `workflows`
Structure des documents de workflow stockés dans MongoDB :
```javascript
{
  _id: ObjectId,
  name: String,
  description: String,
  owner: ObjectId,
  status: String,    // CREATED, VALIDATED, SUBMITTED, SPLITTING, RUNNING, etc.
  created_at: Date,
  updated_at: Date,
  priority: Number,
  estimated_resources: {
    cpu: Number,
    memory: Number,
    storage: Number
  },
  tags: [String],
  metadata: Object
}
```

### Collection `tasks`
Structure des documents de tâches :
```javascript
{
  _id: ObjectId,
  workflow_id: ObjectId,
  name: String,
  command: String,
  parameters: Array,
  dependencies: Array,
  status: String,
  parent_task_id: ObjectId,
  is_subtask: Boolean,
  progress: Number,
  required_resources: Object,
  assigned_to: String,
  results: Object
}
```

## Découpage du projet

### Phase 1 : Analyse et conception (2 semaines)
- Semaine 1 : Étude des workflows et cas d'usage
- Semaine 2 : Spécifications techniques détaillées

### Phase 2 : Développement et intégration (4 semaines)
- Semaine 3 : Module de création des workflows
- Semaine 4 : Algorithmes d'ordonnancement et découpage
- Semaine 5 : Communication avec le coordinateur
- Semaine 6 : Tests d'intégration

### Phase 3 : Optimisation et finalisation (2 semaines)
- Semaine 7 : Optimisation des algorithmes
- Semaine 8 : Documentation et déploiement

## Configuration et démarrage

### Prérequis
- Python 3.10+
- Node.js 18+
- MongoDB 5+
- Docker et Docker Compose

### Installation

#### Configuration manuelle
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# Frontend
cd frontend
npm install
npm start
```

#### Avec Docker
```bash
docker-compose up -d
```

## Équipe

Développé par le Groupe B dans le cadre du projet de système de calcul distribué volontaire uy1.

## Licence

Ce projet est sous licence MIT.# manager-app-2025
