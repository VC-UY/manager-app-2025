import random
from tasks.models import Task, TaskStatus
from workflows.models import Workflow, WorkflowStatus
from volunteers.models import Volunteer, VolunteerTask
from django.utils import timezone
from collections import defaultdict
import logging
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tasks.grok.volunteer_computing_env import VolunteerSchedulingEnv, Task as EnvTask, Workflow as EnvWorkflow, Volunteer as EnvVolunteer
from tasks.grok.train_a3c import ActorCriticNet


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def fcfs_algorithm(tasks: list, volunteers: list) -> dict:
    """
    Algorithme FCFS pour attribuer des tâches aux volontaires.
    
    Args:
        tasks: Liste de dictionnaires contenant les tâches avec leurs exigences
        volunteers: Liste de dictionnaires contenant les volontaires avec leurs ressources et performances
    
    Returns:
        Dictionnaire des assignations {volunteer_id: [(task_id, task_name)]}
    """
    assignments = defaultdict(list)
    volunteer_resources = {
        v["volunteer_id"]: {
            "available_cpu": v["resources"]["cpu_cores"],
            "available_ram": v["resources"]["memory_mb"],
            "available_disk": v["resources"]["disk_space_mb"] / 1024,
            "available_gpu": v["resources"]["gpu"],
            "trust_score": v["performance"]["trust_score"]
        }
        for v in volunteers
    }

    sorted_tasks = sorted(tasks, key=lambda x: x["created_at"])

    volunteer_iter = iter(volunteer_resources.keys())
    for task in sorted_tasks:
        while True:
            try:
                v_id = next(volunteer_iter)
            except StopIteration:
                # Recommencer depuis le début des volontaires
                volunteer_iter = iter(volunteer_resources.keys())
                v_id = next(volunteer_iter)

            # Assigner la tâche au volontaire actuel sans vérifier les ressources
            assignments[v_id].append((task["task_id"], task["task_name"]))
            break

    return dict(assignments)

def mean_execution_time_algorithm(tasks: list, volunteers: list) -> dict:
    """
    Algorithme round robin avec tri initial des volontaires par taux de complétion.
    
    Args:
        tasks: Liste de dictionnaires contenant les tâches avec leurs exigences
        volunteers: Liste de dictionnaires contenant les volontaires avec leurs ressources et performances
    
    Returns:
        Dictionnaire des assignations {volunteer_id: [(task_id, task_name)]}
    """
    assignments = defaultdict(list)

    # Trier les volontaires par completion_rate décroissant
    sorted_volunteers = sorted(
        volunteers,
        key=lambda x: x["performance"]["completion_rate"],
        reverse=True
    )

    # Créer un itérateur cyclique sur les volontaires triés
    volunteer_iter = iter(sorted_volunteers)
    current_volunteer = None

    # Trier les tâches par date de création
    sorted_tasks = sorted(tasks, key=lambda x: x["created_at"])

    logger.info(f"Volontaires disponibles: {volunteers}")
    logger.info(f"Tâches à assigner: {sorted_tasks}")

    for task in sorted_tasks:
        try:
            current_volunteer = next(volunteer_iter)
        except StopIteration:
            # Recommencer depuis le début des volontaires
            volunteer_iter = iter(sorted_volunteers)
            current_volunteer = next(volunteer_iter)

        v_id = current_volunteer["volunteer_id"]
        assignments[v_id].append((task["task_id"], task["task_name"]))

    return dict(assignments)

def round_robin_threshold_algorithm(tasks: list, volunteers: list, trust_threshold: float = 0.5) -> dict:
    """
    Algorithme Round Robin avec seuil de fiabilité. Assigne les tâches en cycle
    parmi les volontaires éligibles (ceux dépassant le seuil de fiabilité).
    
    Args:
        tasks: Liste de dictionnaires contenant les tâches avec leurs exigences
        volunteers: Liste de dictionnaires contenant les volontaires avec leurs ressources et performances
        trust_threshold: Seuil minimal de trust_score pour être éligible
    
    Returns:
        Dictionnaire des assignations {volunteer_id: [(task_id, task_name)]}
    """
    assignments = defaultdict(list)
    volunteer_resources = {
        v["volunteer_id"]: {
            "available_cpu": v["resources"]["cpu_cores"],
            "available_ram": v["resources"]["memory_mb"],
            "available_disk": v["resources"]["disk_space_mb"] / 1024,
            "available_gpu": v["resources"]["gpu"],
            "trust_score": v["performance"]["trust_score"]
        }
        for v in volunteers
    }

    sorted_tasks = sorted(tasks, key=lambda x: x["created_at"])
    eligible_volunteers = [
        v for v in volunteers if v["performance"]["trust_score"] >= trust_threshold
    ]
    
    if not eligible_volunteers:
        logger.warning("Aucun volontaire ne satisfait le seuil de fiabilité")
        return {}

    # Index pour le tourniquet
    current_volunteer_idx = 0

    for task in sorted_tasks:
        required = task["required_resources"]
        required_cpu = required.get("cpu", required.get("cpu_cores", 1))
        required_ram = required.get("ram", required.get("memory_mb", 512))
        required_disk = required.get("disk", required.get("disk_gb", 1))
        required_gpu = required.get("gpu", False)

        # Essayer d'assigner la tâche à un volontaire éligible
        for _ in range(len(eligible_volunteers)):
            volunteer = eligible_volunteers[current_volunteer_idx]
            v_id = volunteer["volunteer_id"]
            res = volunteer_resources[v_id]

            if (
                res["available_cpu"] >= required_cpu and
                res["available_ram"] >= required_ram and
                res["available_disk"] >= required_disk and
                res["available_gpu"] >= required_gpu
            ):
                assignments[v_id].append((task["task_id"], task["task_name"]))
                res["available_cpu"] -= required_cpu
                res["available_ram"] -= required_ram
                res["available_disk"] -= required_disk
                res["available_gpu"] = False if required_gpu else res["available_gpu"]
                break

            # Passer au volontaire suivant (cycle Round Robin)
            current_volunteer_idx = (current_volunteer_idx + 1) % len(eligible_volunteers)

    return dict(assignments)



def random_assigment_algorithm( tasks: list, volunteers: list) -> dict:
    """
    Algorithme aléatoire pour assigner des tâches aux volontaires.
    
    Args:
        tasks: Liste de dictionnaires contenant les tâches avec leurs exigences
        volunteers: Liste de dictionnaires contenant les volontaires avec leurs ressources et performances
    
    Returns:
        Dictionnaire des assignations {volunteer_id: [(task_id, task_name)]}
    """
    assignments = defaultdict(list)
    
    for task in tasks:
        v_id = random.choice(volunteers)["volunteer_id"]
        assignments[v_id].append((task["task_id"], task["task_name"]))
    
    return dict(assignments)





# Chemin du modèle
MODEL_PATH = "/home/sergeo/Master-II/Recherches/Projet_M_I/Groupe B  ManagerApp/v2/manager_backend/tasks/grok/tasks/grok/models/a3c_model_checkpoint_0.pth"

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticNet, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.actor = nn.Linear(32, action_dim)
        self.critic = nn.Linear(32, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.size(-1) < self.input_dim:
            x = torch.cat([x, torch.zeros(x.size(0), self.input_dim - x.size(-1)).to(x.device)], dim=-1)
        elif x.size(-1) > self.input_dim:
            x = x[:, :self.input_dim]
        if torch.isnan(x).any():
            logger.error(f"NaN détecté dans l'entrée : {x}")
        x = self.shared(x)
        if torch.isnan(x).any():
            logger.error(f"NaN détecté dans la sortie partagée : {x}")
        policy_logits = self.actor(x)
        policy_logits = torch.clamp(policy_logits, -3, 3)
        policy_logits = F.softmax(policy_logits, dim=-1)
        if torch.isnan(policy_logits).any():
            logger.error(f"NaN détecté dans policy_logits : {policy_logits}")
        value = self.critic(x)
        return policy_logits, value

def load_model(model, model_path, expected_input_dim, expected_action_dim):
    """
    Charge un modèle pré-entraîné avec ajustement des dimensions dynamiques.
    """
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model_state_dict = model.state_dict()
        for key in state_dict:
            if key in model_state_dict:
                if state_dict[key].shape != model_state_dict[key].shape:
                    logger.warning(f"Dimension incohérente pour {key}: sauvegardé {state_dict[key].shape}, "
                                   f"attendu {model_state_dict[key].shape}. Ajustement.")
                    # Conserver les poids compatibles
                    if key == 'shared.0.weight' and state_dict[key].shape[1] > model_state_dict[key].shape[1]:
                        model_state_dict[key][:, :model_state_dict[key].shape[1]] = state_dict[key][:, :model_state_dict[key].shape[1]]
                    elif key == 'actor.weight' and state_dict[key].shape[0] > model_state_dict[key].shape[0]:
                        model_state_dict[key][:, :] = state_dict[key][:model_state_dict[key].shape[0], :]
                    elif key == 'actor.bias' and state_dict[key].shape[0] > model_state_dict[key].shape[0]:
                        model_state_dict[key][:] = state_dict[key][:model_state_dict[key].shape[0]]
                    else:
                        model_state_dict[key].copy_(state_dict[key])
                else:
                    model_state_dict[key].copy_(state_dict[key])
        model.load_state_dict(model_state_dict)
        logger.info(f"Modèle chargé depuis {model_path} avec ajustements")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}. Initialisation d'un nouveau modèle.")
        # Initialiser un nouveau modèle si le chargement échoue
        model = ActorCriticNet(expected_input_dim, expected_action_dim)
        logger.info("Nouveau modèle A3C initialisé")
    return model

def a3c_algorithm(tasks: list, volunteers: list, model_path: str = MODEL_PATH) -> dict:
    """
    Algorithme A3C pour assigner chaque tâche à un volontaire.
    """

    # Configuration
    MAX_TASKS = 500  # Capacité maximale
    MAX_VOLUNTEERS = 100  # Capacité maximale
    EXPECTED_INPUT_DIM = MAX_TASKS + MAX_VOLUNTEERS + 2  # 602
    EXPECTED_ACTION_DIM = MAX_TASKS * MAX_VOLUNTEERS  # 50000

    if len(tasks) > MAX_TASKS or len(volunteers) > MAX_VOLUNTEERS:
        logger.error(f"Nombre de tâches ({len(tasks)}) ou volontaires ({len(volunteers)}) dépasse les limites maximales")
        return {}

    # Log des ressources
    for t in tasks:
        logger.debug(f"Tâche {t['task_id']}: cpu={t['required_resources'].get('cpu', 1)}, "
                     f"ram_mb={t['required_resources'].get('ram', 512)}, "
                     f"disk_gb={t['required_resources'].get('disk', 1)}")
    for v in volunteers:
        logger.debug(f"Volontaire {v['volunteer_id']}: cpu_cores={v['resources']['cpu_cores']}, "
                     f"memory_go={v['resources']['memory_mb']}, "
                     f"disk_space_mb={v['resources']['disk_space_mb']}")

    # Convertir les données
    env_tasks = [
        EnvTask(
            id=t["task_id"],
            size_in=t["required_resources"].get("disk", 0.5) * 500 / 1000,  # GB → MB, divisé en size_in/size_out
            size_out=t["required_resources"].get("disk", 0.5) * 500 / 1000,
            cpu_req=t["required_resources"].get("cpu", 0.5),
            mem_req=t["required_resources"].get("ram", 256) / 1024,  # MB → GB
            duration_est=1.0,
            task_type="generic"
        )
        for t in tasks
    ]
    env_volunteers = [
        EnvVolunteer(
            id=v["volunteer_id"],
            cpu=v["resources"]["cpu_cores"],
            mem=v["resources"]["memory_mb"],  # Go (déjà correct)
            disk=v["resources"]["disk_space_mb"] / 1000,  # Mo → Go
            uplink=100.0,
            downlink=100.0,
            reliability=v["performance"]["trust_score"] / 100.0,  # Normalisé
            vol_type="generic"
        )
        for v in volunteers
    ]
    env_workflow = EnvWorkflow(id="temp_wf", tasks=env_tasks)
    env = VolunteerSchedulingEnv(env_workflow, env_volunteers, max_tasks=MAX_TASKS, max_volunteers=MAX_VOLUNTEERS)

    # Vérifier les dimensions réelles
    actual_input_dim = env.observation_space['shape'][0]  # 602 avec padding
    actual_action_dim = env.action_space['n']  # 50000
    logger.info(f"Dimensions environnement : input={actual_input_dim}, action={actual_action_dim}")

    # Charger le modèle avec ajustement
    model = ActorCriticNet(input_dim=actual_input_dim, action_dim=actual_action_dim)
    model = load_model(model, model_path, actual_input_dim, actual_action_dim)

    assignments = defaultdict(list)
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    done = False
    max_steps = len(tasks) * len(volunteers) * 2

    step = 0
    while not done and step < max_steps:
        valid_actions = env.info.get('valid_actions', list(range(actual_action_dim)))
        if not valid_actions:
            logger.warning(f"Étape {step} : Aucune action valide disponible")
            break
        with torch.no_grad():
            policy_logits, _ = model(state)
            if policy_logits.size(-1) != actual_action_dim:
                logger.error(f"Dimension de policy_logits incorrecte : {policy_logits.size(-1)} au lieu de {actual_action_dim}")
                break
            mask = torch.zeros(actual_action_dim)
            for idx in valid_actions:
                if idx < actual_action_dim:
                    mask[idx] = 1
            masked_logits = policy_logits * mask
            masked_logits = masked_logits / (masked_logits.sum(dim=-1, keepdim=True) + 1e-10)
            dist = Categorical(probs=masked_logits)
            action = dist.sample().item()

        next_state, reward, done, info = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        state = next_state
        step += 1

        logger.debug(f"Étape {step}: action={action}, reward={reward}, done={done}, info={info}")

        # Récupérer les assignations en utilisant les IDs directement
        for v_name, task_names in info["task_assignments"].items():
            try:
                v_id = v_name.replace("Volunteer_", "")
                if not any(v["volunteer_id"] == v_id for v in volunteers):
                    logger.warning(f"Volontaire ID {v_id} non trouvé dans volunteers")
                    continue
                for task_name in task_names:
                    t_id = task_name.replace("Task_", "")
                    task = next((t for t in tasks if t["task_id"] == t_id), None)
                    if not task:
                        logger.warning(f"Tâche ID {t_id} non trouvée dans tasks")
                        continue
                    t_name = task["task_name"]
                    if (t_id, t_name) not in assignments[v_id]:
                        assignments[v_id].append((t_id, t_name))
                        logger.debug(f"Tâche {t_id} ({t_name}) assignée au volontaire {v_id}")
            except Exception as e:
                logger.warning(f"Erreur lors de la récupération de l'assignation pour v_name={v_name}: {e}")
                continue

    # Repli : assigner les tâches restantes aléatoirement
    if len(env.assigned_tasks) < len(tasks):
        logger.warning(f"Seulement {len(env.assigned_tasks)}/{len(tasks)} tâches assignées. Assignation aléatoire des restantes.")
        remaining_tasks = [t for t in env_tasks if t.id not in env.assigned_tasks]
        for task in remaining_tasks:
            valid_volunteers = [
                v for v in env_volunteers
                if (v.cpu >= task.cpu_req and
                    v.mem >= task.mem_req and
                    v.disk >= task.size_in + task.size_out)
            ]
            if valid_volunteers:
                vol = random.choice(valid_volunteers)
                v_id = next(v["volunteer_id"] for v in volunteers if v["volunteer_id"] == str(vol.id))
                t_id = task.id
                t_name = task.name
                assignments[v_id].append((t_id, t_name))
                env.assigned_tasks.add(t_id)
                logger.debug(f"Tâche restante {t_id} assignée au volontaire {v_id}")
            else:
                logger.warning(f"Aucun volontaire valide pour la tâche restante {task.id}")

    logger.info(f"Assignations A3C terminées après {step} étapes. Tâches assignées : {len(env.assigned_tasks)}/{len(tasks)}")
    return dict(assignments)

def assign_workflow_to_volunteers(workflow: Workflow, volunteers_data: list, algorithm: str = "a3c") -> dict:
    """
    Assigne les tâches d'un workflow aux volontaires selon l'algorithme spécifié.
    
    Args:
        workflow: Instance du workflow
        volunteers_data: Liste des données des volontaires
        algorithm: Algorithme à utiliser ("fcfs", "mean_execution_time", "round_robin", "a3c")
    
    Returns:
        Dictionnaire des assignations
    """
    logger.info(f"Début de l'assignation du workflow {workflow.id} avec l'algorithme {algorithm}")
    
    # Créer ou mettre à jour les volontaires
    volunteer_objs = []
    for vdata in volunteers_data:
        try:
            volunteer_id = vdata.get("volunteer_id")
            if not volunteer_id:
                logger.error(f"Données de volontaire sans ID: {vdata}")
                continue
                
            resources = vdata.get("resources")
            v, created = Volunteer.objects.update_or_create(
                coordinator_volunteer_id=volunteer_id,
                defaults={
                    "name": vdata.get("username", f"Volontaire {volunteer_id}"),
                    "cpu_cores": resources.get("cpu_cores", 1),
                    "ram_mb": resources.get("memory_mb", 1024),
                    "disk_gb": int(resources.get("disk_space_mb", 10240) / 1024),
                    "status": "available",
                    "gpu": resources.get("gpu", False),
                    "ip_address": resources.get("ip_address", '0.0.0.0'),
                    "meta_info" : {"performance": vdata.get("performance", 10240)}
                }
            )
            volunteer_objs.append(v)
            logger.info(f"Volontaire {'créé' if created else 'mis à jour'}: {v.name}")
        except Exception as e:
            logger.error(f"Erreur lors de la création/mise à jour du volontaire {volunteer_id}: {e}")
            continue
    
    tasks = workflow.tasks.filter(status=TaskStatus.CREATED).order_by('created_at')
    tasks_data = [
        {
            "task_id": str(t.id),
            "task_name": t.name,
            "required_resources": t.required_resources,
            "created_at": t.created_at
        }
        for t in tasks
    ]
    
    # Sélectionner l'algorithme
    if algorithm == "mean_execution_time":
        assignments = mean_execution_time_algorithm(tasks_data, volunteers_data)
    elif algorithm == "round_robin":
        assignments = round_robin_threshold_algorithm(tasks_data, volunteers_data, trust_threshold=0.5)
    elif algorithm == "a3c":
        assignments = a3c_algorithm(tasks_data, volunteers_data, model_path=MODEL_PATH)
    elif algorithm == 'random_assignment':
        assignments = random_assigment_algorithm(tasks_data, volunteers_data)               
    else:  # Par défaut, FCFS
        assignments = fcfs_algorithm(tasks_data, volunteers_data)
    
    
    task_assignments = defaultdict(list)
    
    for v_id, tasks_list in assignments.items():
        volunteer = Volunteer.objects.get(coordinator_volunteer_id=v_id)
        for task_id, task_name in tasks_list:
            task = Task.objects.get(id=task_id)
            volunteer_task = VolunteerTask.objects.create(
                volunteer=volunteer,
                task=task,
                assigned_at=timezone.now(),
                status=TaskStatus.ASSIGNED
            )
            task.status = TaskStatus.ASSIGNED
            task.save()
            task_assignments[v_id].append({
                "task_id": str(task.id),
                "task_name": task.name,
                "assignment_id": str(volunteer_task.id)
            })
        
        volunteer_resources = next(v for v in volunteers_data if v["volunteer_id"] == v_id)
        if (
            volunteer_resources["resources"]["cpu_cores"] == 0 or
            volunteer_resources["resources"]["memory_mb"] == 0 or
            volunteer_resources["resources"]["disk_space_mb"] == 0
        ):
            volunteer.status = "busy"
            volunteer.save()
    
    assigned_count = workflow.tasks.filter(status=TaskStatus.ASSIGNED).count()
    if assigned_count > 0:
        workflow.status = WorkflowStatus.PENDING
        workflow.save()
    
    logger.warning(f"Résumé des assignations: {len(task_assignments)} volontaires, {assigned_count}/{tasks.count()} tâches assignées")
    
    return dict(task_assignments)

def create_task_handler(task_id: str, volunteer_id: str):
    """
    Crée un handler pour une tâche assignée à un volontaire.
    
    Args:
        task_id: ID de la tâche
        volunteer_id: ID du volontaire
    
    Returns:
        Dictionnaire contenant les informations du handler
    """
    try:
        task = Task.objects.get(id=task_id)
        volunteer = Volunteer.objects.get(coordinator_volunteer_id=volunteer_id)
        
        volunteer_task = VolunteerTask.objects.create(
            volunteer=volunteer,
            task=task,
            assigned_at=timezone.now(),
            status=TaskStatus.ASSIGNED
        )
        
        logger.info(f"Tâche {task.name} assignée au volontaire {volunteer.name}")
        
        return {
            "task_id": str(task.id),
            "volunteer_id": str(volunteer.id),
            "assignment_id": str(volunteer_task.id),
            "status": "ASSIGNED"
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création du handler pour la tâche {task_id}: {e}")
        return None