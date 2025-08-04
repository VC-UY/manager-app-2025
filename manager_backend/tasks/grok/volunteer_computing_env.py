import numpy as np
import random
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class Task:
    def __init__(self, id, size_in, size_out, cpu_req, mem_req, duration_est, task_type, preferences=None):
        self.id = id
        self.size_in = size_in
        self.size_out = size_out
        self.cpu_req = cpu_req
        self.mem_req = mem_req
        self.duration_est = duration_est
        self.task_type = task_type
        self.preferences = preferences or {}
        self.status = 'pending'
        self.start_time = None
        self.end_time = None
        self.name = f"Task_{id}"

class Volunteer:
    def __init__(self, id, cpu, mem, disk, uplink, downlink, reliability, vol_type):
        self.id = id
        self.cpu = cpu
        self.mem = mem
        self.disk = disk
        self.uplink = uplink
        self.downlink = downlink
        self.reliability = reliability
        self.vol_type = vol_type
        self.cpu_used = 0
        self.mem_used = 0
        self.disk_used = 0
        self.status = 'available'
        self.execution_count = 0
        self.success_rate = 1.0
        self.name = f"Volunteer_{id}"

class Workflow:
    def __init__(self, id, tasks, priority=1.0, preferences=None):
        self.id = id
        self.tasks = {t.id: t for t in tasks}
        self.status = 'pending'
        self.priority = priority
        self.preferences = preferences or {}

class VolunteerSchedulingEnv:
    def __init__(self, workflow, volunteers, w1=100.0, w2=10.0, w3=0.001, w4=200.0, max_tasks=500, max_volunteers=100):
        self.workflow = workflow
        self.volunteers = {v.id: v for v in volunteers}
        self.task_ids = list(self.workflow.tasks.keys())
        self.vol_ids = list(self.volunteers.keys())
        self.task_assignments = defaultdict(list)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.makespan = 0.0
        self.failures = 0
        self.total_data_transferred = 0.0
        self.step_count = 0
        self.max_steps = len(self.task_ids) * 10
        self.max_tasks = max_tasks
        self.max_volunteers = max_volunteers
        # Dimensions dynamiques avec padding jusqu'aux maximums
        self.observation_space = {'shape': (self.max_tasks + self.max_volunteers + 2,)}  # 602
        self.action_space = {'n': self.max_tasks * self.max_volunteers}  # 50000
        self.assigned_tasks = set()
        self.available_task_ids = self.task_ids.copy()
        self.current_time = 0.0
        self.info = {}

    def _get_state(self):
        task_states = [1.0 if t.status == 'pending' else 0.0 for t in self.workflow.tasks.values()]
        task_states += [0.0] * (self.max_tasks - len(task_states))  # Padding jusqu'à 500
        volunteer_states = [v.cpu / (v.cpu or 1) for v in self.volunteers.values()]
        volunteer_states += [0.0] * (self.max_volunteers - len(volunteer_states))  # Padding jusqu'à 100
        workflow_state = [self.workflow.priority, 1.0 if self.workflow.status == 'in_progress' else 0.0]
        state = np.array(task_states + volunteer_states + workflow_state, dtype=np.float32)
        if np.isnan(state).any():
            logger.error(f"NaN détecté dans l'état : {state}")
        return state

    def _check_preferences(self, task, volunteer):
        return True

    def _compute_resource_utilization(self):
        total_util = 0.0
        for v in self.volunteers.values():
            util = v.execution_count / (len(self.task_ids) or 1)
            total_util += util
        return total_util / len(self.volunteers) if self.volunteers else 0.0

    def reset(self):
        self.task_assignments = defaultdict(list)
        self.makespan = 0.0
        self.failures = 0
        self.total_data_transferred = 0.0
        self.step_count = 0
        self.current_time = 0.0
        for t in self.workflow.tasks.values():
            t.status = 'pending'
            t.start_time = None
            t.end_time = None
        for v in self.volunteers.values():
            v.cpu_used = 0
            v.mem_used = 0
            v.disk_used = 0
            v.status = 'available'
            v.execution_count = 0
            v.success_rate = 1.0
        self.assigned_tasks = set()
        self.available_task_ids = self.task_ids.copy()
        logger.debug(f"Réinitialisation : {len(self.available_task_ids)} tâches disponibles")
        self.info = {'valid_actions': self._get_valid_actions()}
        return self._get_state()

    def _get_valid_actions(self):
        valid_actions = []
        num_tasks = len(self.available_task_ids)
        logger.debug(f"Calcul des actions valides pour {num_tasks} tâches restantes")
        for t_idx in range(num_tasks):
            task_id = self.available_task_ids[t_idx]
            task = self.workflow.tasks[task_id]
            for v_idx, vol_id in enumerate(self.vol_ids):
                vol = self.volunteers[vol_id]
                action = t_idx * self.max_volunteers + v_idx  # Utiliser max_volunteers pour mapper
                if (vol.cpu >= task.cpu_req and
                    vol.mem >= task.mem_req and
                    vol.disk >= task.size_in + task.size_out):
                    valid_actions.append(action)
                else:
                    logger.debug(f"Action invalide : tâche {task_id}, volontaire {vol_id}, "
                                 f"cpu_requis={task.cpu_req}, cpu_dispo={vol.cpu}, "
                                 f"mem_requis={task.mem_req}, mem_dispo={vol.mem}, "
                                 f"disk_requis={task.size_in + task.size_out}, disk_dispo={vol.disk}")
        if not valid_actions:
            logger.warning(f"Aucune action valide disponible. Tâches restantes : {len(self.available_task_ids)}")
        return valid_actions

    def step(self, action):
        reward = 0.0
        max_assign = self.max_tasks * self.max_volunteers

        if action < max_assign and self.available_task_ids and action < self.action_space['n']:
            t_idx = action // self.max_volunteers
            if t_idx < len(self.available_task_ids):
                task_id = self.available_task_ids[t_idx]
                v_idx = action % self.max_volunteers
                if v_idx < len(self.vol_ids):  # Vérifier que l'index est dans les volontaires réels
                    vol_id = self.vol_ids[v_idx]
                    task = self.workflow.tasks[task_id]
                    vol = self.volunteers[vol_id]

                    if (vol.cpu >= task.cpu_req and
                        vol.mem >= task.mem_req and
                        vol.disk >= task.size_in + task.size_out):
                        task.start_time = self.current_time
                        task.end_time = self.current_time + task.duration_est
                        self.current_time += task.duration_est
                        self.makespan = max(self.makespan, task.end_time)
                        logger.debug(f"Tâche {task_id} assignée au volontaire {vol_id}, "
                                     f"début={task.start_time}, fin={task.end_time}, makespan={self.makespan}")

                        self.task_assignments[vol.name].append(task.name)
                        self.assigned_tasks.add(task_id)
                        self.available_task_ids.pop(t_idx)
                        task.status = 'done'
                        self.total_data_transferred += task.size_in + task.size_out

                        reward += self.w1 * (1 / (task.duration_est or 1e-6)) * self.workflow.priority
                        util = task.cpu_req / (vol.cpu or 1)
                        reward += self.w4 * util * vol.reliability
                        reward -= self.w3 * (task.size_in + task.size_out)

                        vol.execution_count += 1
                        vol.success_rate = 1.0
                    else:
                        reward -= self.w2
                        self.failures += 1
                        logger.debug(f"Échec d'assignation : tâche {task_id}, volontaire {vol_id}, "
                                     f"cpu_requis={task.cpu_req}, cpu_dispo={vol.cpu}, "
                                     f"mem_requis={task.mem_req}, mem_dispo={vol.mem}")
            else:
                reward -= self.w2
                self.failures += 1
        else:
            reward -= self.w2
            self.failures += 1
            logger.warning(f"Action invalide reçue : {action}, max_assign={max_assign}")

        self.workflow.status = 'completed' if len(self.assigned_tasks) == len(self.task_ids) else 'in_progress'
        self.step_count += 1
        done = len(self.assigned_tasks) == len(self.task_ids)

        if self.workflow.status == 'completed':
            reward += self.w1 * len(self.task_ids)

        self.info = {
            'makespan': self.makespan,
            'failures': self.failures,
            'data_transferred': self.total_data_transferred,
            'resource_utilization': self._compute_resource_utilization(),
            'task_assignments': dict(self.task_assignments),
            'valid_actions': self._get_valid_actions()
        }

        return self._get_state(), reward, done, self.info