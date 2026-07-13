# Guide d'utilisation — Système d'apprentissage distribué sur machines volontaires

> **Projet de Mémoire Master II — MBASSI ATANGANA Yannick Serge**
> *Conception d'un framework frugal d'apprentissage distribué sur machines volontaires*

## Table des matières

1. [Architecture & Fonctionnalités Clés](#architecture--fonctionnalités-clés)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Test local (1 machine)](#test-local-1-machine)
5. [Démarrage pas à pas (multi-machines)](#démarrage-pas-à-pas)
6. [Déploiement automatique SSH](#déploiement-automatique-ssh)
7. [Configuration avancée](#configuration-avancée)
8. [Monitoring en temps réel](#monitoring-en-temps-réel)
9. [Comprendre les statistiques](#comprendre-les-statistiques)
10. [Scénarios d'expérience recommandés (Thèmes de recherche)](#scénarios-dexpérience-recommandés-thèmes-de-recherche)
11. [Dépannage](#dépannage)

---

## Architecture & Fonctionnalités Clés

```
┌──────────────────────────────────────────────────────────────┐
│                        Machine A                             │
│                     COORDINATEUR :9000                       │
│  • Reçoit les heartbeats des volontaires                     │
│  • Maintient la liste des nœuds actifs                       │
│  • Gère la détection de défaillance (heartbeats, timeouts)   │
│  • Diffuse cette liste au Manager toutes les 5 s             │
└────────────────────────┬─────────────────────────────────────┘
                         │ MSG_VOLUNTEER_LIST (TCP)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                        Machine B                             │
│                       MANAGER :9001                          │
│  • Reçoit la liste des volontaires                           │
│  • Calcule les k voisins XOR (Kademlia-style)                │
│  • Assigne dynamiquement les rôles AD-PSGD (Actif/Passif)    │
│  • Route les modèles compressés entre volontaires            │
│  • Publie les statistiques globales                          │
└────────────────────────┬─────────────────────────────────────┘
                         │ Échanges de modèles (TCP, via files)
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Volontaire 0   Volontaire 1   Volontaire N
     Machine C      Machine D      Machine …
     • Heartbeat → Coordinateur
     • Entraînement local sécurisé (SGD avec rollback sur NaN/Inf)
     • Push modèle → Manager → pair
     • Pull modèles reçus ← Manager
     • Sélection adaptative des pairs (SW-UCB avec exploration)
     • Compression frugale (JointSQ / Quantization / Sparsification)
     • Agrégation AD-PSGD (Bipartite, averaging symétrique)
     • Ajustement du LR (AdaLoss / AdaStair)
     • Profilage système bas niveau (/proc, /sys, perf stat, throttling)
```

### 1. Topologie XOR & Peer Sampling SW-UCB
* **Topologie XOR** : L'adresse IP de chaque volontaire est hachée (SHA-256 → uint64). La distance entre deux volontaires est le XOR de leurs hachés.
* **SW-UCB Selector (Garivier & Moulines, 2008)** : Chaque nœud choisit ses pairs à l'aide d'un bandit manchot glissant. La récompense composite équilibre la bande passante, la latence, le succès de transfert, un **bonus de diversité** (anti-monopole) et un retour d'**utilité du modèle** (gain d'accuracy post-agrégation) pour s'adapter aux réseaux hétérogènes et aux données Non-IID.
* **Exploration $\epsilon$-greedy** : Un taux d'exploration $\epsilon$ dynamique (décroissant au fil des rounds) force périodiquement la sélection de pairs sous-explorés pour découvrir de nouvelles routes.

### 2. Optimisation AD-PSGD (Lian et al., 2018)
* **Topologies de Ring** : `ring` (voisins directs) ou `exponential` (sauts de pas $2^k$).
* **Évitement de Deadlocks** : Partitionnement bipartite en rôles **Actif** (initie l'averaging) et **Passif** (répond uniquement aux sollicitations).
* **Adaptive Skipping** : Les nœuds passifs sautent dynamiquement les étapes d'averaging si leur *staleness* (distance de modèle $\||x - \hat{x}\||_2$) est inférieure à un seuil critique, économisant le trafic réseau.

### 3. Compression Frugale JointSQ
* **JointSQ** : Algorithme conjoint de quantification et sparsification basé sur l'optimisation MCKP (Multiple-Choice Knapsack Problem) gloutonne pour maximiser la précision transmise sous contrainte stricte de bande passante.
* **Méthodes alternatives** : Quantification uniforme (`quantization` 8-bits) ou sparsification top-k (`sparsification`).

### 4. Ajustement Adaptatif du Taux d'Apprentissage
* **AdaStair** : Décroissance par paliers à des rounds spécifiques définis.
* **AdaLoss** : Suivi de la stagnation de la loss de validation et division automatique du taux d'apprentissage par 2 lorsque le plateau est atteint.

### 5. Profilage Matériel Avancé (Edge AI)
* Analyse continue par lecture directe de `/proc/[pid]/status` et `/sys/devices/system/cpu` :
  - **Memory (RSS, PSS, USS)** pour isoler le coût marginal réel du modèle et détecter les fuites.
  - **Thermal Throttling & Fréquences CPU** pour identifier les ralentissements d'origine matérielle.
  - **IPC (Instructions Par Cycle)** via `perf stat` pour séparer les inefficacités logicielles des limites physiques.

---

## Prérequis

| Composant | Version minimale |
|-----------|-----------------|
| Python    | 3.9             |
| PyTorch   | 2.0.0 (CPU/CUDA)|
| torchvision | 0.15.0        |
| numpy     | 1.24.0          |
| psutil    | 5.9.0           |
| OS        | Linux (recommandé pour le profilage avancé), macOS, Windows |

*Note sur le profilage* : Les fonctionnalités de profilage avancées (PSS/USS, fréquences CPU exactes, IPC via perf) nécessitent un noyau **Linux** et les privilèges d'accès adéquats (ex: `/proc/sys/kernel/perf_event_paranoid` configuré). Des solutions de repli (fallbacks) automatiques via `psutil` sont implémentées pour macOS/Windows.

---

## Installation

### Sur chaque machine (Coordinateur, Manager, Volontaires)

```bash
# 1. Cloner ou copier le projet
git clone <repo>
cd distributed_learning_2/distributed_learning

# 2. Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Test local (1 machine)

Le script `launch_experiment.py` configure et orchestre une simulation complète en local. Il simule des adresses IPs distinctes (`10.0.0.1`, `10.0.0.2`...) pour garantir le bon calcul de la topologie XOR.

### Exemple de commandes d'expérimentation

```bash
# Lancement classique (AD-PSGD activé, compression JointSQ, LR AdaLoss)
python launch_experiment.py --n-volunteers 3 --dataset cifar10 --max-rounds 15

# Comparer sans AD-PSGD (Gossip standard FedAvg) et sans compression
python launch_experiment.py --n-volunteers 4 --compression none --adpsgd-enabled false

# Expérimenter avec sparsification pure et topologie Ring
python launch_experiment.py --n-volunteers 5 --compression sparsification --sparsity 0.05 --adpsgd-topology ring

# Utiliser le scheduler AdaStair avec 4 volontaires en mode Non-IID
python launch_experiment.py --n-volunteers 4 --partition non-iid --adaptive-lr adastair --max-rounds 20
```

### Arguments de `launch_experiment.py`

| Argument | Choix / Type | Défaut | Description |
|----------|--------------|--------|-------------|
| `--n-volunteers` | `int` | `3` | Nombre de nœuds volontaires |
| `--model` | `resnet18`, `resnet50`, `resnet101`, `resnet152`, `vgg19` | `resnet18` | Modèle de deep learning |
| `--dataset` | `cifar10`, `cifar100`, `imagenet` | `cifar10` | Base de données d'apprentissage |
| `--partition` | `iid`, `non-iid` | `iid` | Mode de répartition des données |
| `--compression` | `quantization`, `sparsification`, `jointsq`, `none` | `jointsq` | Méthode de réduction de taille |
| `--sparsity` | `float` | `0.05` | Ratio de paramètres conservés pour top-k (5%) |
| `--bits` | `int` | `8` | Bits pour la quantification (int8) |
| `--adaptive-lr` | `none`, `adastair`, `adaloss` | `adaloss` | Scheduler de taux d'apprentissage |
| `--adpsgd-enabled` | `true`, `false` | `true` | Activer l'algorithme AD-PSGD |
| `--adpsgd-topology`| `ring`, `exponential` | `exponential` | Topologie logique d'échange |
| `--max-rounds` | `int` | `15` | Nombre max de rounds de communication |

---

## Démarrage pas à pas

### Étape 1 — Lancer le Manager (Machine B)
```bash
export MANAGER_HOST=192.168.1.130
export MANAGER_PORT=9001
export K_NEIGHBORS=3
python3 manager.py
```

### Étape 2 — Lancer le Coordinateur (Machine A)
```bash
export COORDINATOR_HOST=192.168.1.120
export COORDINATOR_PORT=9000
export MANAGER_EXTERNAL_HOST=192.168.1.130
export MANAGER_PORT=9001
python3 coordinator.py
```

### Étape 3 — Lancer chaque Volontaire (Machines distantes)
```bash
# Machine C (ID 0)
python3 volunteer.py --id 0 --n-volunteers 3 --coordinator 192.168.1.120 --manager 192.168.1.130

# Machine D (ID 1)
python3 volunteer.py --id 1 --n-volunteers 3 --coordinator 192.168.1.120 --manager 192.168.1.130
```

---

## Déploiement automatique SSH

Utilisez le script `deploy.sh` configuré pour votre parc de machines volontaires :
1. Éditez les variables `COORD_IP`, `MANAGER_IP` et `VOL_IPS` à l'intérieur du script.
2. Lancez le déploiement en une seule commande :
   ```bash
   ./deploy.sh
   ```

---

## Configuration avancée

Toutes les variables d'environnement suivantes sont prises en compte au démarrage des nœuds :

* `ADPSGD_ENABLED` (`true`|`false`) : Active AD-PSGD.
* `ADPSGD_TOPOLOGY` (`ring`|`exponential`) : Choix de la topologie de réseau.
* `ADPSGD_SKIP_FACTOR_MAX` (Défaut: `5`) : Limite haute du facteur de skipping adaptatif.
* `ADPSGD_STALENESS_THRESHOLD` (Défaut: `0.05`) : Seuil L2 en dessous duquel l'averaging est sauté.
* `ADAPTIVE_LR_METHOD` (`none`|`adastair`|`adaloss`) : Politique d'ajustement du LR.
* `COMPRESSION` (`none`|`quantization`|`sparsification`|`jointsq`) : Méthode de compression.

---

## Monitoring en temps réel

Lancez le tableau de bord de monitoring pour visualiser la progression en direct (ex: précision de test, taux de compression, octets routés) :
```bash
python3 monitor.py --manager 127.0.0.1 --interval 5
```

---

## Comprendre les statistiques

Chaque expérience génère des fichiers JSON détaillés dans `results/` :
- `global_stats.json` : Résumé global du manager (trafic total routé, débit, liste des volontaires actifs).
- `volunteer_<ip_simulee>.json` : Fichier de stats par volontaire contenant un tableau `rounds` avec toutes les mesures prises au cours de l'entraînement.

### Métriques clés à observer dans le JSON du volontaire :

```json
{
  "round_num": 3,
  "train_loss": 0.412,
  "test_acc": 0.784,
  "learning_rate": 0.0005,
  "compression_ratio": 12.4,
  "batch_time_avg_s": 0.045,
  
  "cpu_percent_peak": 88.5,
  "cpu_percent_mean": 42.1,
  "ram_usage_gb_peak": 1.25,
  "throttle_ratio": 0.12,  // 12% de throttling thermique détecté
  "ipc": 1.15,             // Instructions par cycle
  
  "adpsgd": {
    "adpsgd_role": "passive",
    "adpsgd_topology": "exponential",
    "adpsgd_staleness_norm": 0.0241,  // Norme L2 de dérive du modèle
    "adpsgd_n_averaging_skip": 1,     // Nombre d'averagings sautés
    "adpsgd_skip_factor": 3           // Dynamic skip factor actuel
  }
}
```

---

## Scénarios d'expérience recommandés (Thèmes de recherche)

Voici 5 protocoles expérimentaux directement exploitables pour la rédaction scientifique de votre mémoire :

### Thème 1 : Topologie et Décentralisation Asynchrone (AD-PSGD)
* **Objectif** : Comparer l'impact de la topologie logique sur la vitesse de diffusion du modèle.
* **Protocole** : 
  1. Lancez une expérience avec `--adpsgd-topology ring` et notez l'évolution du score `test_acc` et du paramètre `adpsgd_staleness_norm` au cours des rounds.
  2. Répétez l'expérience avec `--adpsgd-topology exponential`.
* **Attente théorique** : La topologie exponentielle présente un gap spectral plus élevé, accélérant la convergence globale au prix d'une connectivité logique plus complexe.

### Thème 2 : Analyse de la Frugalité Réseau (JointSQ)
* **Objectif** : Évaluer l'efficacité de la compression MCKP (JointSQ) face aux méthodes de base.
* **Protocole** :
  1. Lancez l'apprentissage avec `--compression none` (mesure témoin de la BW totale).
  2. Testez avec `--compression quantization --bits 8`.
  3. Testez avec `--compression jointsq`.
* **Attente théorique** : JointSQ doit démontrer un meilleur compromis (Pareto optimal) entre le ratio de compression réel et la perte de précision induite sur le modèle final.

### Thème 3 : Robustesse et Bandits Contextuels (SW-UCB)
* **Objectif** : Prouver que l'ajout du signal de diversité et de qualité de modèle empêche la monopolisation des nœuds rapides.
* **Protocole** :
  1. Observez la répartition des sélections dans `volunteer_<ip>.json` (champ `neighbors_info`).
  2. Modifiez artificiellement le fichier `src/peer_sampling.py` pour couper le bonus de diversité ou le feedback de modèle (ex: mettre leur poids à 0).
* **Attente théorique** : Sans ces bonus, le système converge vers une exploitation pure du nœud possédant le meilleur lien réseau (monopole), alors que le système complet maintient une couverture large du réseau, indispensable en cas de données Non-IID.

### Thème 4 : Mitigation du Drifting sous Données Non-IID
* **Objectif** : Montrer comment les algorithmes d'ajustement du LR stabilisent l'apprentissage asynchrone lorsque les distributions de données divergent.
* **Protocole** :
  1. Lancez `--partition non-iid --adaptive-lr none`.
  2. Lancez `--partition non-iid --adaptive-lr adaloss`.
* **Attente théorique** : AdaLoss amortit les oscillations provoquées par le drift de modèle en divisant localement le pas d'apprentissage lorsque la convergence s'essouffle.

### Thème 5 : Profilage Matériel et Stragglers sur l'Edge
* **Objectif** : Identifier et corréler les goulets d'étranglement physiques.
* **Protocole** :
  1. Examinez les variables `throttle_ratio` et `ipc` récoltées sur les différents nœuds.
  2. Si un nœud ralentit le round (`ete_seconds` élevé), déterminez si le blocage provient de ressources saturées (IPC bas, RAM saturée) ou d'un bridage thermique (`throttle_ratio` élevé).
* **Attente théorique** : Le profiler avancé permet de distinguer de façon déterministe un bug logiciel (ex: fuite de mémoire) d'une contrainte d'environnement physique.

---

*Framework d'Apprentissage Distribué Frugal — Master II*
