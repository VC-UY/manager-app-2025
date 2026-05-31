# Guide d'utilisation — Système d'apprentissage distribué sur machines volontaires

> **Projet de Mémoire Master II — MBASSI ATANGANA Yannick Serge**
> *Conception d'un framework frugal d'apprentissage distribué sur machines volontaires*

## Table des matières

1. [Architecture](#architecture)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Test local (1 machine)](#test-local-1-machine)
5. [Démarrage pas à pas (multi-machines)](#démarrage-pas-à-pas)
6. [Déploiement automatique SSH](#déploiement-automatique-ssh)
7. [Configuration avancée](#configuration-avancée)
8. [Monitoring en temps réel](#monitoring-en-temps-réel)
9. [Comprendre les statistiques](#comprendre-les-statistiques)
10. [Scénarios d'expérience recommandés](#scénarios-dexpérience-recommandés)
11. [Dépannage](#dépannage)
8. [Dépannage](#dépannage)
9. [Scénarios d'expérience recommandés](#scénarios-dexpérience-recommandés)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Machine A                             │
│                     COORDINATEUR :9000                       │
│  • Reçoit les heartbeats des volontaires                     │
│  • Maintient la liste des nœuds actifs                       │
│  • Diffuse cette liste au Manager toutes les 5 s             │
└────────────────────────┬─────────────────────────────────────┘
                         │ MSG_VOLUNTEER_LIST (TCP)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                        Machine B                             │
│                       MANAGER :9001                          │
│  • Reçoit la liste des volontaires                           │
│  • Calcule les k voisins XOR (Kademlia-style)                │
│  • Route les modèles compressés entre volontaires            │
│  • Publie les statistiques globales                          │
└────────────────────────┬─────────────────────────────────────┘
                         │ Échanges de modèles (TCP, via files)
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Volontaire 0   Volontaire 1   Volontaire N
     Machine C      Machine D      Machine …
     • Heartbeat → Coordinateur
     • Entraînement local (SGD)
     • Push modèle → Manager → pair
     • Pull modèles reçus ← Manager
     • Agrégation FedAvg
     • Statistiques par round
```

**Topologie XOR** : L'adresse IP de chaque volontaire est hachée (SHA-256 → uint64).
La distance entre deux volontaires est le XOR de leurs hachés.
Les k voisins les plus proches (petite distance XOR) forment la vue locale du nœud.

**Gossip learning** : À chaque round, un volontaire envoie son modèle compressé
à `GOSSIP_FANOUT` voisins aléatoires parmi ses k plus proches, puis récupère
les modèles que ses pairs lui ont envoyés, et effectue une moyenne FedAvg.

---

## Prérequis

| Composant | Version minimale |
|-----------|-----------------|
| Python    | 3.9             |
| PyTorch   | 2.0             |
| torchvision | 0.15          |
| numpy     | 1.24            |
| OS        | Linux / macOS / Windows |

**Réseau** : Le coordinateur et le manager doivent être joignables depuis tous les
volontaires. Les volontaires communiquent uniquement via le manager (pas de P2P direct).

**Ports à ouvrir dans le pare-feu** :
- Machine coordinateur : TCP entrant 9000
- Machine manager : TCP entrant 9001

---

## Installation

### Sur chaque machine (coordinateur, manager, volontaires)

```bash
# 1. Cloner / copier le projet
git clone <repo>   # ou copier le dossier distributed_learning/
cd distributed_learning

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Optionnel GPU) Installer PyTorch CUDA
# Voir https://pytorch.org/get-started/locally/ pour la commande adaptée
```

---

## Test local (1 machine)

Le script `launch_experiment.py` démarre le manager, le coordinateur et N volontaires
en sous-processus sur une seule machine, avec des IPs simulées (ex: `10.0.0.1`,
`10.0.0.2` …) pour activer la topologie XOR réelle même en local.

```bash
# Lancement minimal (3 volontaires, MNIST, non-IID, quantification)
python launch_experiment.py --n-volunteers 3

# Paramétrage complet
python launch_experiment.py \
    --n-volunteers 5       \   # nombre de volontaires
    --dataset mnist        \   # mnist | cifar10
    --partition non-iid    \   # iid | non-iid
    --compression quantization \  # quantization | sparsification | none
    --k 3                  \   # voisins XOR par nœud
    --gossip-interval 30   \   # secondes entre rounds
    --local-epochs 3           # epochs locaux par round

# Les logs sont dans ./logs/   (un fichier par composant)
# Les stats sont dans ./results/

# Monitoring en direct (dans un autre terminal)
python monitor.py
```

> **Conseil** : commencez toujours par un test local avant le déploiement multi-machines.
> Vérifiez que `results/global_stats.json` se remplit bien après quelques minutes.

---

## Démarrage pas à pas

### Étape 1 — Démarrer le Manager (machine B)

```bash
# Variables d'environnement (optionnelles, les valeurs par défaut sont dans src/config.py)
export MANAGER_HOST=0.0.0.0
export MANAGER_PORT=9001
export K_NEIGHBORS=4
export STATS_PRINT_INTERVAL=30

python3 manager.py
```

> Démarrez **toujours le manager en premier**, car le coordinateur
> essaie de lui envoyer la liste dès le premier volontaire connecté.

---

### Étape 2 — Démarrer le Coordinateur (machine A)

```bash
export COORDINATOR_HOST=0.0.0.0
export COORDINATOR_PORT=9000
export MANAGER_EXTERNAL_HOST=<IP_PUBLIQUE_MACHINE_B>
export MANAGER_PORT=9001
export HEARTBEAT_TIMEOUT=35

python3 coordinator.py
```

---

### Étape 3 — Démarrer chaque Volontaire (machines C, D, …)

```bash
# Arguments obligatoires :
#   --id          : identifiant unique 0-indexé de ce volontaire
#   --n-volunteers: nombre total de volontaires prévus
#   --coordinator : IP publique du coordinateur
#   --manager     : IP publique du manager

# Volontaire 0 (sur machine C)
python3 volunteer.py \
    --id 0
    --n-volunteers 5 \
    --coordinator 192.168.1.106 \
    --manager     192.168.1.130

# Volontaire 1 (sur machine D)
python3 volunteer.py \
    --id 1 \
    --n-volunteers 5 \
    --coordinator 192.168.1.106 \
    --manager     192.168.1.106
```

L'argument `--my-ip` est optionnel : si omis, le volontaire détecte automatiquement
son IP sortante en tentant de se connecter au coordinateur.

---

## Configuration avancée

Toutes les variables d'environnement suivantes peuvent être définies avant le lancement.

### Paramètres d'entraînement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `DATASET` | `mnist` | Dataset : `mnist` ou `cifar10` |
| `DATA_PARTITION` | `iid` | Partition des données : `iid` ou `non-iid` |
| `LOCAL_EPOCHS` | `3` | Epochs d'entraînement local par round |
| `BATCH_SIZE` | `32` | Taille de batch |
| `LEARNING_RATE` | `0.01` | Taux d'apprentissage SGD |
| `NUM_CLASSES` | `10` | Nombre de classes |

### Paramètres gossip / topologie

| Variable | Défaut | Description |
|----------|--------|-------------|
| `K_NEIGHBORS` | `4` | Nombre de voisins XOR par nœud |
| `GOSSIP_INTERVAL` | `60` | Secondes entre deux rounds gossip |
| `GOSSIP_FANOUT` | `2` | Voisins contactés par round |

### Paramètres de compression (frugalité bande passante)

| Variable | Défaut | Description |
|----------|--------|-------------|
| `COMPRESSION` | `quantization` | Méthode : `quantization`, `sparsification`, `none` |
| `QUANTIZATION_BITS` | `8` | Bits pour la quantification (8 = int8) |
| `SPARSIFICATION_RATIO` | `0.05` | Fraction des paramètres conservés (top-5 %) |

**Économies typiques** :
- `quantization` (int8) : ~4× moins de bande passante (float32→int8)
- `sparsification` (top-5 %) : ~15–20× selon le modèle
- Combinaison possible en chaînant les méthodes dans `compression.py`

### Robustesse réseau

| Variable | Défaut | Description |
|----------|--------|-------------|
| `HEARTBEAT_INTERVAL` | `10` | Secondes entre heartbeats |
| `HEARTBEAT_TIMEOUT` | `35` | Secondes d'inactivité avant expulsion |
| `SOCKET_TIMEOUT` | `60` | Timeout de toute connexion TCP |
| `MAX_RETRIES` | `3` | Tentatives avant abandon |
| `RETRY_DELAY` | `5.0` | Pause en secondes entre tentatives |

---

## Déploiement automatique SSH

Le script `deploy.sh` automatise le déploiement complet sur plusieurs machines via SSH.

### Pré-requis SSH

```bash
# Configurer l'accès SSH sans mot de passe (clés) vers toutes les machines
ssh-keygen -t ed25519                          # générer une paire de clés si inexistante
ssh-copy-id ubuntu@192.168.1.10               # coordinateur
ssh-copy-id ubuntu@192.168.1.11               # manager
ssh-copy-id ubuntu@192.168.1.20               # volontaire 0
# etc.

# Copier le projet sur toutes les machines
for IP in 192.168.1.10 192.168.1.11 192.168.1.20 192.168.1.21 192.168.1.22; do
    rsync -av --exclude '__pycache__' --exclude 'venv' \
        ./distributed_learning/ ubuntu@$IP:~/distributed_learning/
done

# Installer les dépendances sur chaque machine
for IP in 192.168.1.10 192.168.1.11 192.168.1.20 192.168.1.21 192.168.1.22; do
    ssh ubuntu@$IP "cd ~/distributed_learning && \
        python3 -m venv venv && \
        venv/bin/pip install -r requirements.txt"
done
```

### Lancement avec deploy.sh

```bash
chmod +x deploy.sh

# Déploiement avec configuration par défaut (modifier les IPs dans le script)
COORD_IP=192.168.1.10 \
MANAGER_IP=192.168.1.11 \
VOL_IPS="192.168.1.20 192.168.1.21 192.168.1.22" \
DATASET=mnist \
DATA_PARTITION=non-iid \
COMPRESSION=quantization \
K_NEIGHBORS=3 \
./deploy.sh
```

Le script :

1. Vérifie la connectivité SSH vers toutes les machines
2. Arrête tout processus existant du projet
3. Crée les dossiers `logs/` et `results/` sur chaque machine
4. Démarre Manager → Coordinateur → Volontaires dans le bon ordre

### Lancement manuel (multi-machines)

Adapter les IPs selon votre réseau (ex: `192.168.1.*` pour réseau local université).

```bash
# ── Machine B : Manager (TOUJOURS EN PREMIER) ──────────────────────────────
MANAGER_HOST=0.0.0.0 \
MANAGER_PORT=9001 \
MANAGER_EXTERNAL_HOST=192.168.1.11 \
K_NEIGHBORS=3 \
STATS_PRINT_INTERVAL=30 \
python manager.py > logs/manager.log 2>&1 &

# ── Machine A : Coordinateur ────────────────────────────────────────────────
COORDINATOR_HOST=0.0.0.0 \
COORDINATOR_PORT=9000 \
MANAGER_EXTERNAL_HOST=192.168.1.11 \
MANAGER_PORT=9001 \
HEARTBEAT_TIMEOUT=35 \
python coordinator.py > logs/coordinator.log 2>&1 &

# ── Machine C : Volontaire 0 ────────────────────────────────────────────────
DATASET=mnist \
DATA_PARTITION=non-iid \
COMPRESSION=quantization \
LOCAL_EPOCHS=3 \
GOSSIP_INTERVAL=60 \
python volunteer.py \
    --id 0 --n-volunteers 5 \
    --coordinator 192.168.1.10 \
    --manager     192.168.1.11 \
    --my-ip       192.168.1.20 \
    > logs/volunteer_0.log 2>&1 &

# Répéter pour les volontaires 1, 2, 3, 4 (--id 1, 2, 3, 4)
```

> **Note NAT / université** : si vos machines sont derrière un NAT, spécifiez toujours
> `--my-ip` avec l'IP visible depuis le réseau local. Le système supporte cette
> configuration : le coordinateur utilise l'IP déclarée dans le heartbeat.

---

## Monitoring en temps réel

### Depuis n'importe quelle machine du réseau

```bash
# Monitoring continu (rafraîchissement toutes les 10s)
python3 monitor.py --manager 192.168.1.11

# Ajuster l'intervalle
python3 monitor.py --manager 192.168.1.11 --interval 5

# Exporter les stats en JSON et quitter
python3 monitor.py --manager 192.168.1.106 --export resultats_finaux.json

# Mode log (sans effacement écran — utile pour redirection fichier)
python3 monitor.py --manager 192.168.1.11 --no-clear >> monitor.log
```

### Consulter les logs

```bash
# Log du manager (échanges de modèles, stats globales)
tail -f logs/manager.log

# Log d'un volontaire (rounds gossip, précision, bande passante)
tail -f logs/volunteer_0.log

# Voir toutes les stats en direct sur le manager
tail -f logs/manager.log | grep "STATISTIQUES\|test_acc\|bytes_routed"
```

### Récupérer les résultats depuis les machines distantes

```bash
# Récupérer les stats de tous les nœuds
for IP in 192.168.1.10 192.168.1.11 192.168.1.20 192.168.1.21 192.168.1.22; do
    rsync -av ubuntu@$IP:~/distributed_learning/results/ ./results_collected/$IP/
done
```

---

## Comprendre les statistiques

### Fichiers produits

```
results/
├── global_stats.json          # Stats globales (manager)
├── volunteer_10_0_0_1.json    # Stats du volontaire 10.0.0.1
├── volunteer_10_0_0_2.json
└── ...
```

### Structure `global_stats.json`

```json
{
  "runtime_s": 3600,
  "n_active_volunteers": 5,
  "total_model_exchanges": 120,
  "total_bytes_routed": 15728640,
  "throughput_KB_per_s": 4.36,
  "volunteer_summaries": {
    "10.0.0.1": {
      "total_rounds": 24,
      "best_test_acc": 0.9721,
      "total_bytes_sent": 3145728
    }
  }
}
```

### Structure `volunteer_<ip>.json`

```json
{
  "volunteer_ip": "10.0.0.1",
  "total_rounds": 24,
  "best_test_acc": 0.9721,
  "final_test_acc": 0.9698,
  "total_train_duration_s": 486.3,
  "total_bytes_sent": 3145728,
  "total_bytes_received": 2097152,
  "avg_compression_ratio": 3.87,
  "rounds": [
    {
      "round_num": 1,
      "train_loss": 0.6821,
      "train_acc": 0.7812,
      "test_acc": 0.8234,
      "train_duration_s": 21.4,
      "bytes_sent": 131072,
      "bytes_received": 131072,
      "n_models_received": 1,
      "compression_ratio": 3.91
    }
  ]
}
```

### Métriques clés

| Métrique | Source | Interprétation |
|----------|--------|----------------|
| `test_acc` | Volontaire | Précision sur le jeu de test complet |
| `compression_ratio` | Volontaire | > 1 = gain, ex. 4.0 = 4× moins de bande passante |
| `throughput_KB_per_s` | Manager | Débit total routé (tous échanges confondus) |
| `total_bytes_routed` | Manager | Bande passante totale consommée côté manager |
| `train_duration_s` | Volontaire | Temps d'entraînement local par round |

---

## Dépannage

### Volontaire ne trouve pas le coordinateur
```
Connexion coordinateur perdue : [Errno 111] Connection refused
```
→ Vérifier que `coordinator.py` est bien démarré et que le port 9000 est ouvert.
→ Vérifier `--coordinator <IP>` pointe vers la bonne machine.

### Manager dit "Destinataire inconnu"
```
Modèle refusé : Destinataire inconnu : 10.0.0.3
```
→ Le coordinateur n'a pas encore transmis la liste au manager.
→ Attendre quelques secondes (délai max 5 s entre diffusions).
→ Vérifier que `MANAGER_EXTERNAL_HOST` sur le coordinateur pointe bien vers le manager.

### Volontaire ne reçoit jamais de modèle
→ Vérifier que `--my-ip` correspond à l'IP visible par les autres machines.
→ Si test local, utiliser des IPs fictives distinctes (`--my-ip 10.0.0.1`, etc.).

### Erreur "Payload trop grand"
→ Réduire `SPARSIFICATION_RATIO` ou utiliser `COMPRESSION=quantization`.
→ Augmenter `MAX_MODEL_BYTES` si le modèle est volontairement grand.

### Consommation mémoire excessive sur le manager
→ La file par volontaire peut grossir si un nœud ne poll pas.
→ Réduire `GOSSIP_INTERVAL` ou augmenter la fréquence de poll dans `volunteer.py`.

---

## Scénarios d'expérience recommandés

### Expérience 1 — Impact de la compression sur la bande passante

Lancer 3 fois la même expérience avec :
```bash
COMPRESSION=none         python volunteer.py …
COMPRESSION=quantization python volunteer.py …
COMPRESSION=sparsification SPARSIFICATION_RATIO=0.05 python volunteer.py …
```
Comparer `total_bytes_sent` et `avg_compression_ratio` dans les fichiers stats.

---

### Expérience 2 — Impact du nombre de voisins (k)

```bash
K_NEIGHBORS=2  # vue très locale
K_NEIGHBORS=4  # défaut
K_NEIGHBORS=8  # vue plus large
```
Mesurer la vitesse de convergence (`test_acc` par round) et la bande passante.

---

### Expérience 3 — Données non-IID vs IID

```bash
DATA_PARTITION=iid     # distribution uniforme
DATA_PARTITION=non-iid # hétérogénéité réaliste (2 classes par volontaire)
```
Observer la convergence et l'écart-type de précision entre volontaires.

---

### Expérience 4 — Robustesse aux déconnexions

1. Lancer 5 volontaires.
2. Après 5 rounds, couper (Ctrl+C) 1 ou 2 volontaires.
3. Observer que le système continue à fonctionner.
4. Redémarrer les volontaires coupés.
5. Vérifier que la précision reprend progressivement.

---

### Expérience 5 — Scalabilité

Augmenter progressivement le nombre de volontaires :
`N = 3, 5, 10, 20`

Mesurer :
- Débit total routé (`throughput_KB_per_s`)
- Temps moyen par round sur le manager
- Précision de convergence

---

*Projet de mémoire — Apprentissage distribué frugal sur machines volontaires*
