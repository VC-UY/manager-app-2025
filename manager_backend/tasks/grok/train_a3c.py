import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
import numpy as np
from tasks.grok.volunteer_computing_env import VolunteerSchedulingEnv, Task, Workflow, Volunteer
# from volunteer_computing_env import VolunteerSchedulingEnv, Task, Workflow, Volunteer
import logging
import os
import random
import pathlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - Worker %(processName)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

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
            logging.error(f"NaN détecté dans l'entrée : {x}")
        x = self.shared(x)
        if torch.isnan(x).any():
            logging.error(f"NaN détecté dans la sortie partagée : {x}")
        policy_logits = self.actor(x)
        policy_logits = torch.clamp(policy_logits, -3, 3)
        policy_logits = F.softmax(policy_logits, dim=-1)
        if torch.isnan(policy_logits).any():
            logging.error(f"NaN détecté dans policy_logits : {policy_logits}")
        value = self.critic(x)
        return policy_logits, value

    def act(self, state, epsilon=0.1, valid_actions=None):
        device = state.device
        if random.random() < epsilon or not valid_actions:
            action = random.choice(valid_actions) if valid_actions else random.randint(0, self.action_dim - 1)
            action_tensor = torch.tensor([action], dtype=torch.long).to(device)
            dist = Categorical(logits=torch.zeros(1, self.action_dim).to(device))
            log_prob = dist.log_prob(action_tensor)
        else:
            policy_logits, _ = self.forward(state)
            mask = torch.zeros(self.action_dim).to(device)
            for idx in valid_actions or []:
                if idx < self.action_dim:
                    mask[idx] = 1
            masked_logits = policy_logits * mask
            masked_logits = masked_logits / (masked_logits.sum(dim=-1, keepdim=True) + 1e-10)
            if torch.isnan(masked_logits).any():
                logging.error(f"NaN détecté dans masked_logits : {masked_logits}")
                masked_logits = torch.where(torch.isnan(masked_logits), torch.zeros_like(masked_logits), masked_logits)
            dist = Categorical(probs=masked_logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
        return action_tensor.item(), log_prob

    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(probs=logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, torch.squeeze(values), entropy

class A3CWorker:
    def __init__(self, wid, env_fn, global_model, optimizer, gamma=0.99, max_steps=500):
        self.wid = wid
        self.env = env_fn()
        self.global_model = global_model
        self.optimizer = optimizer
        self.gamma = gamma
        self.max_steps = max_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model = ActorCriticNet(
            input_dim=self.env.observation_space['shape'][0],
            action_dim=self.env.action_space['n']
        ).to(self.device)
        self.local_model.load_state_dict(global_model.state_dict())
        self.logger = logging.getLogger(f'Worker_{wid}')
        self.episode_count = 0
        self.update_count = 0
        self.model_dir = pathlib.Path("tasks/grok/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        self.logger.info(f"Démarrage de l'entraînement pour le worker {self.wid}")
        episode_rewards = []
        episode_makespans = []
        episode_failures = []
        episode_assigned = []
        epsilon = 0.9
        epsilon_decay = 0.995
        min_epsilon = 0.1
        try:
            for episode in range(2000):
                state = self.env.reset()
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                done = False
                episode_reward = 0.0
                step = 0
                values, log_probs, rewards, entropies = [], [], [], []

                self.logger.info(f"Worker {self.wid}, Épisode {episode}: Début, epsilon={epsilon:.3f}")

                while not done and step < self.max_steps:
                    valid_actions = self.env.info.get('valid_actions', list(range(self.env.action_space['n'])))
                    if not valid_actions:
                        self.logger.warning(f"Aucune action valide à l'étape {step}, épisode {episode}")
                        reward = -self.env.w2
                        self.env.failures += 1
                        self.env.step_count += 1
                        self.env.info['valid_actions'] = self.env._get_valid_actions()
                        rewards.append(reward)
                        continue
                    action, log_prob = self.local_model.act(state, epsilon, valid_actions)
                    if action >= self.env.action_space['n']:
                        self.logger.error(f"Action invalide générée : {action}, max={self.env.action_space['n'] - 1}")
                        action = random.choice(valid_actions)  # Revenir à une action valide
                    next_state, reward, done, info = self.env.step(action)
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    self.env.info = info

                    values.append(self.local_model(state)[1])
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    entropies.append(Categorical(probs=self.local_model(state)[0]).entropy())
                    episode_reward += reward

                    state = next_state
                    step += 1

                    if done or step % 5 == 0:
                        self._update_global(next_state, done, values, log_probs, rewards, entropies)
                        self.local_model.load_state_dict(self.global_model.state_dict())
                        values, log_probs, rewards, entropies = [], [], [], []

                episode_rewards.append(episode_reward)
                episode_makespans.append(info['makespan'])
                episode_failures.append(info['failures'])
                episode_assigned.append(len(self.env.assigned_tasks))
                if episode % 50 == 0:
                    self.logger.info(
                        f"Worker {self.wid}, Épisode {episode}, "
                        f"Récompense: {np.mean(episode_rewards[-50:]):.2f}, "
                        f"Makespan: {np.mean(episode_makespans[-50:]):.2f}, "
                        f"Échecs: {np.mean(episode_failures[-50:]):.2f}, "
                        f"Tâches assignées: {np.mean(episode_assigned[-50:]):.2f}"
                    )
                    model_path = self.model_dir / f"a3c_model_checkpoint_{self.wid}.pth"
                    torch.save(self.global_model.state_dict(), model_path)
                    self.logger.debug(f"Worker {self.wid}: Checkpoint sauvegardé à {model_path}")
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                self.episode_count += 1
        except Exception as e:
            self.logger.error(f"Erreur dans le worker {self.wid}: {str(e)}")
        finally:
            self.logger.info(f"Worker {self.wid} terminé")

    def _update_global(self, next_state, done, values, log_probs, rewards, entropies):
        R = 0 if done else self.local_model(next_state)[1].item()
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        values = torch.cat(values)
        for i, lp in enumerate(log_probs):
            if lp.dim() != 1 or lp.size(0) != 1:
                self.logger.error(f"Taille de log_prob incohérente à l'index {i}: {lp.size()}")
                log_probs[i] = lp.squeeze()
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        advantage = returns - values.squeeze()
        actor_loss = -(log_probs.squeeze() * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_bonus = entropies.mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 0.3)
        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            if local_param.grad is not None:
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad += local_param.grad
        self.optimizer.step()

        self.update_count += 1
        self.logger.debug(f"Worker {self.wid}: Mise à jour globale terminée, Perte={loss.item():.2f}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # Fixer les nombres de tâches et volontaires à 500 et 100 pour l'entraînement
    num_tasks = 500
    num_volunteers = 100
    tasks = [Task(id=f"task_{i}", size_in=0.5, size_out=0.5,  # 1 GB total
                  cpu_req=1.0, mem_req=0.512,  # 512 MB
                  duration_est=np.random.uniform(0.5, 1), task_type="generic",
                  preferences={"vol_type": "generic"})
             for i in range(num_tasks)]
    workflow = Workflow(id="wf_1", tasks=tasks, priority=0.8, preferences={"vol_type": "generic"})
    volunteers = [Volunteer(id=i, cpu=4.0, mem=np.random.uniform(3, 15),  # 3-15 Go
                           disk=np.random.uniform(100000, 300000),  # 100-300 Go
                           uplink=20, downlink=20, reliability=np.random.uniform(0.9, 1.0),
                           vol_type="generic")
                 for i in range(num_volunteers)]
    env_fn = lambda: VolunteerSchedulingEnv(workflow, volunteers)

    env = env_fn()
    input_dim = env.observation_space['shape'][0]  # Doit être 602 (500 + 100 + 2)
    action_dim = env.action_space['n']  # Doit être 50000 (500 * 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = ActorCriticNet(input_dim, action_dim).to(device)
    global_model.share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-5)

    logging.info(f"Démarrage de l'entraînement avec 1 worker, input_dim={input_dim}, action_dim={action_dim}")
    processes = []
    for wid in range(1):
        worker = A3CWorker(wid, env_fn, global_model, optimizer)
        p = mp.Process(target=worker.run)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    try:
        model_path = pathlib.Path("tasks/grok/models/a3c_model_final.pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_model.state_dict(), model_path)
        logging.info(f"Entraînement terminé, modèle sauvegardé à {model_path}")
    except Exception as e:
        logging.error(f"Échec de la sauvegarde du modèle final à {model_path}: {str(e)}")