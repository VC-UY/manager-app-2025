#backend/tasks/services/task_assigner.py

import logging
import requests
import json
from django.utils import timezone
from django.conf import settings
from tasks.models import Task, TaskStatus

logger = logging.getLogger(__name__)

def assign_task(task, volunteer_id=None):
    """
    Attribue une tâche à un volontaire spécifique ou au meilleur volontaire disponible.
    Optimisé pour les tâches matricielles (addition et multiplication).
    
    Args:
        task (Task): La tâche à attribuer
        volunteer_id (str, optional): ID du volontaire à qui attribuer la tâche
        
    Returns:
        bool: True si la tâche a été attribuée, False sinon
    """
    logger.info(f"Attribution de la tâche {task.id}")
    
    if task.status != TaskStatus.PENDING:
        logger.warning(f"La tâche {task.id} ne peut pas être attribuée car son statut est {task.status}")
        return False
    
    # Si un volontaire spécifique est demandé
    if volunteer_id:
        # Vérifier si le volontaire est disponible via l'API du coordinateur
        if not is_volunteer_available(volunteer_id):
            logger.warning(f"Le volontaire {volunteer_id} n'est pas disponible")
            return False
        
        assigned_to = volunteer_id
    else:
        # Trouver le meilleur volontaire disponible
        assigned_to = find_best_volunteer_for_matrix_task(task)
        
        if not assigned_to:
            logger.warning(f"Aucun volontaire disponible pour la tâche {task.id}")
            return False
    
    # Mettre à jour la tâche
    task.assigned_to = assigned_to
    task.status = TaskStatus.RUNNING
    task.start_time = timezone.now()
    task.save()
    
    logger.info(f"Tâche {task.id} attribuée à {assigned_to}")
    
    # Notifier le volontaire via le système de message Pub/Sub
    notify_result = notify_volunteer(task, assigned_to)
    
    if not notify_result:
        logger.error(f"Échec de la notification au volontaire {assigned_to} pour la tâche {task.id}")
        # Marquer la tâche comme en attente à nouveau
        task.assigned_to = None
        task.status = TaskStatus.PENDING
        task.start_time = None
        task.save()
        return False
    
    return True

def find_best_volunteer_for_matrix_task(task):
    """
    Trouve le meilleur volontaire pour une tâche matricielle.
    Implémente les algorithmes 2 ou 5 du document selon le type de tâche.
    """
    # Récupérer les volontaires disponibles via l'API du coordinateur
    available_volunteers = get_available_volunteers()
    
    if not available_volunteers:
        return None
    
    # Récupérer les exigences de ressources
    resource_requirements = task.required_resources or {}
    
    # Trier les volontaires selon différentes stratégies en fonction du type de tâche
    workflow_type = task.workflow.workflow_type
    
    if workflow_type in ['MATRIX_ADDITION', 'Addition de matrices']:
        # Pour l'addition, prioriser la mémoire
        sorted_volunteers = sort_volunteers_for_addition(available_volunteers, resource_requirements)
    elif workflow_type in ['MATRIX_MULTIPLICATION', 'Multiplication de matrices']:
        # Pour la multiplication, prioriser CPU/GPU
        sorted_volunteers = sort_volunteers_for_multiplication(available_volunteers, resource_requirements)
    else:
        # Tri générique pour les autres types de tâches
        sorted_volunteers = sort_volunteers_by_overall_score(available_volunteers, resource_requirements)
    
    # Sélectionner le meilleur volontaire
    if sorted_volunteers and len(sorted_volunteers) > 0:
        return sorted_volunteers[0]['id']
    
    return None

def sort_volunteers_for_addition(volunteers, requirements):
    """
    Trie les volontaires pour une tâche d'addition matricielle.
    Pour l'addition, la mémoire est le facteur le plus important.
    """
    # Calculer un score pour chaque volontaire
    for volunteer in volunteers:
        resources = volunteer.get('resources', {})
        
        # Score de base
        score = 0
        
        # 1. Vérifier la mémoire (facteur principal pour l'addition)
        mem_req = normalize_memory_requirement(requirements.get('memory', '256MB'))
        mem_avail = normalize_memory_requirement(resources.get('memory', '1GB'))
        
        if mem_avail >= mem_req:
            score += 10
            # Bonus pour surplus de mémoire (jusqu'à +5 points)
            score += min(5, (mem_avail - mem_req) / (mem_req * 0.5))
        else:
            # Pénalité sévère si mémoire insuffisante
            score -= 20
        
        # 2. Vérifier le CPU
        cpu_req = normalize_cpu_requirement(requirements.get('cpu', 'low'))
        cpu_avail = normalize_cpu_requirement(resources.get('cpu', 'medium'))
        
        if cpu_avail >= cpu_req:
            score += 5
            # Petit bonus pour surplus de CPU
            score += min(3, (cpu_avail - cpu_req))
        else:
            # Pénalité modérée pour CPU insuffisant
            score -= 5
        
        # 3. Considérer l'historique des performances
        perf_history = volunteer.get('performanceHistory', {})
        if 'matrix_add_block' in perf_history:
            success_rate = perf_history['matrix_add_block'].get('successRate', 0.8)
            score *= (0.5 + success_rate/2)  # Facteur entre 0.5 et 1
        
        volunteer['score'] = score
    
    # Trier par score décroissant et retourner la liste triée
    return sorted(volunteers, key=lambda v: v.get('score', 0), reverse=True)

def sort_volunteers_for_multiplication(volunteers, requirements):
    """
    Trie les volontaires pour une tâche de multiplication matricielle.
    Pour la multiplication, le CPU/GPU est le facteur le plus important.
    """
    # Calculer un score pour chaque volontaire
    for volunteer in volunteers:
        resources = volunteer.get('resources', {})
        
        # Score de base
        score = 0
        
        # 1. Vérifier le CPU (facteur principal pour la multiplication)
        cpu_req = normalize_cpu_requirement(requirements.get('cpu', 'medium'))
        cpu_avail = normalize_cpu_requirement(resources.get('cpu', 'medium'))
        
        if cpu_avail >= cpu_req:
            score += 10
            # Bonus pour surplus de CPU (jusqu'à +5 points)
            score += min(5, (cpu_avail - cpu_req) * 2)
        else:
            # Pénalité sévère si CPU insuffisant
            score -= 15
        
        # 2. Vérifier le GPU si requis ou préféré
        if 'gpu' in requirements:
            if requirements['gpu'] == 'required' and not resources.get('gpu', False):
                # GPU requis mais non disponible
                score -= 30  # Forte pénalité
            elif requirements['gpu'] == 'preferred':
                if resources.get('gpu', False):
                    score += 15  # Bonus important pour GPU préféré
        elif resources.get('gpu', False):
            # GPU disponible même si non explicitement requis
            score += 5  # Petit bonus
        
        # 3. Vérifier la mémoire
        mem_req = normalize_memory_requirement(requirements.get('memory', '512MB'))
        mem_avail = normalize_memory_requirement(resources.get('memory', '1GB'))
        
        if mem_avail >= mem_req:
            score += 8
            # Bonus pour surplus de mémoire
            score += min(4, (mem_avail - mem_req) / (mem_req * 0.5))
        else:
            # Pénalité pour mémoire insuffisante
            score -= 10
        
        # 4. Considérer l'historique des performances
        perf_history = volunteer.get('performanceHistory', {})
        if 'matrix_multiply_block' in perf_history:
            success_rate = perf_history['matrix_multiply_block'].get('successRate', 0.8)
            score *= (0.5 + success_rate/2)  # Facteur entre 0.5 et 1
        
        # 5. Bonus pour volontaires avec bonne vitesse de réseau (important pour multiplication)
        network_quality = resources.get('network', 'medium')
        if network_quality == 'high':
            score += 5
        elif network_quality == 'very_high':
            score += 8
        
        volunteer['score'] = score
    
    # Trier par score décroissant et retourner la liste triée
    return sorted(volunteers, key=lambda v: v.get('score', 0), reverse=True)

def sort_volunteers_by_overall_score(volunteers, requirements):
    """
    Trie les volontaires selon un score global équilibré.
    Utilisé pour les types de tâches génériques.
    """
    # Calculer un score pour chaque volontaire
    for volunteer in volunteers:
        resources = volunteer.get('resources', {})
        
        # Score de base
        score = 0
        
        # CPU
        cpu_req = normalize_cpu_requirement(requirements.get('cpu', 'medium'))
        cpu_avail = normalize_cpu_requirement(resources.get('cpu', 'medium'))
        
        if cpu_avail >= cpu_req:
            score += 10
            # Bonus pour surplus
            score += min(5, (cpu_avail - cpu_req) * 1.5)
        else:
            score -= 10
        
        # Mémoire
        mem_req = normalize_memory_requirement(requirements.get('memory', '512MB'))
        mem_avail = normalize_memory_requirement(resources.get('memory', '1GB'))
        
        if mem_avail >= mem_req:
            score += 10
            # Bonus pour surplus
            score += min(5, (mem_avail - mem_req) / (mem_req * 0.5))
        else:
            score -= 10
        
        # GPU
        if 'gpu' in requirements:
            if requirements['gpu'] == 'required' and not resources.get('gpu', False):
                score -= 20
            elif requirements['gpu'] == 'preferred':
                if resources.get('gpu', False):
                    score += 10
        
        # Réseau
        net_req = normalize_network_requirement(requirements.get('network', 'medium'))
        net_avail = normalize_network_requirement(resources.get('network', 'medium'))
        net_score = compare_network_capability(net_avail, net_req)
        score += net_score
        
        # Historique des performances (générique)
        perf_history = volunteer.get('performanceHistory', {})
        avg_success_rate = 0.8  # Valeur par défaut
        if perf_history:
            rates = [data.get('successRate', 0.8) for data in perf_history.values()]
            if rates:
                avg_success_rate = sum(rates) / len(rates)
        
        score *= (0.5 + avg_success_rate/2)
        
        volunteer['score'] = score
    
    # Trier par score décroissant et retourner la liste triée
    return sorted(volunteers, key=lambda v: v.get('score', 0), reverse=True)

# Fonctions utilitaires

def normalize_cpu_requirement(req):
    """Normalise les exigences CPU"""
    if isinstance(req, (int, float)):
        return float(req)
    cpu_values = {"low": 1.0, "medium": 2.0, "high": 4.0, "very_high": 8.0}
    return cpu_values.get(req, 2.0)

def normalize_memory_requirement(req):
    """Normalise les exigences mémoire en MB"""
    if isinstance(req, (int, float)):
        return float(req)
    if isinstance(req, str):
        if "GB" in req:
            value = req.replace("GB", "").strip()
            try:
                return float(value) * 1024
            except ValueError:
                pass
        elif "MB" in req:
            value = req.replace("MB", "").strip()
            try:
                return float(value)
            except ValueError:
                pass
    
    mem_values = {"low": 256.0, "medium": 512.0, "high": 1024.0, "very_high": 2048.0}
    return mem_values.get(req, 512.0)

def normalize_network_requirement(req):
    """Normalise les exigences réseau"""
    if isinstance(req, str):
        return req.lower()
    return "medium"

def compare_network_capability(available, required):
    """Compare les capacités réseau et retourne un score"""
    net_levels = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
    avail_level = net_levels.get(available, 2)
    req_level = net_levels.get(required, 2)
    
    if avail_level >= req_level:
        score = 5
        # Bonus pour capacité excédentaire
        score += min(5, (avail_level - req_level) * 2)
        return score
    else:
        # Pénalité si en dessous des exigences
        return -5 * (req_level - avail_level)

# Fonctions d'interaction avec le coordinateur

def get_available_volunteers():
    """
    Récupère la liste des volontaires disponibles via l'API du coordinateur
    """
    try:
        # URL de l'API du coordinateur
        coordinator_api = settings.COORDINATOR_API_URL
        volunteers_endpoint = f"{coordinator_api}/volunteers/available"
        
        # Faire la requête
        response = requests.get(
            volunteers_endpoint,
            headers={"Content-Type": "application/json"},
            timeout=5  # 5 secondes de timeout
        )
        
        response.raise_for_status()  # Lever une exception en cas d'erreur HTTP
        
        volunteers_data = response.json()
        
        # Valider le format des données
        if not isinstance(volunteers_data, list):
            logger.error("Format de données invalide reçu de l'API des volontaires")
            return []
        
        return volunteers_data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la récupération des volontaires disponibles: {e}")
        return []
    
    except ValueError as e:
        logger.error(f"Erreur lors du parsing de la réponse JSON: {e}")
        return []

def is_volunteer_available(volunteer_id):
    """
    Vérifie si un volontaire spécifique est disponible via l'API du coordinateur
    """
    try:
        # URL de l'API du coordinateur
        coordinator_api = settings.COORDINATOR_API_URL
        volunteer_endpoint = f"{coordinator_api}/volunteers/{volunteer_id}/status"
        
        # Faire la requête
        response = requests.get(
            volunteer_endpoint,
            headers={"Content-Type": "application/json"},
            timeout=5  # 5 secondes de timeout
        )
        
        response.raise_for_status()  # Lever une exception en cas d'erreur HTTP
        
        status_data = response.json()
        
        return status_data.get('status') == 'available'
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la vérification de la disponibilité du volontaire {volunteer_id}: {e}")
        return False
    
    except ValueError as e:
        logger.error(f"Erreur lors du parsing de la réponse JSON: {e}")
        return False

def notify_volunteer(task, volunteer_id):
    """
    Notifie un volontaire de l'attribution d'une tâche via le broker de messages
    """
    try:
        # URL du broker de messages
        broker_url = settings.MESSAGE_BROKER_URL
        notification_endpoint = f"{broker_url}/publish"
        
        # Préparer le message
        message = {
            "topic": "tasks/assignment",
            "payload": {
                "task_id": str(task.id),
                "volunteer_id": volunteer_id,
                "command": task.command,
                "parameters": task.parameters,
                "resources": task.required_resources,
                "docker_image": task.docker_image,
                "matrix_block": {
                    "row_start": task.matrix_block_row_start,
                    "row_end": task.matrix_block_row_end,
                    "col_start": task.matrix_block_col_start,
                    "col_end": task.matrix_block_col_end
                } if None not in [task.matrix_block_row_start, task.matrix_block_row_end,
                                 task.matrix_block_col_start, task.matrix_block_col_end] else None,
                "timestamp": timezone.now().isoformat()
            }
        }
        
        # Envoi du message
        response = requests.post(
            notification_endpoint,
            headers={"Content-Type": "application/json"},
            json=message,
            timeout=5  # 5 secondes de timeout
        )
        
        response.raise_for_status()  # Lever une exception en cas d'erreur HTTP
        
        # Vérifier la réponse
        result = response.json()
        
        return result.get('success', False)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la notification du volontaire {volunteer_id}: {e}")
        return False
    
    except ValueError as e:
        logger.error(f"Erreur lors du parsing de la réponse JSON: {e}")
        return False