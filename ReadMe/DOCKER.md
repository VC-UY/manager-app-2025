# Docker Deployment — Distributed Learning Volunteer Node

> **Companion to [USAGE.md](USAGE.md)** — this document covers the Docker-specific files,
> their design rationale, and how they improve the experimental workflow.
> For architecture, configuration, and troubleshooting, see USAGE.md.
> To publish the image to Docker Hub so volunteers can pull it, see
> **[PUBLISH.md](PUBLISH.md)**.

## Table of contents

1. [Why Docker?](#why-docker)
2. [Files overview](#files-overview)
3. [Design decisions](#design-decisions)
4. [Quick start](#quick-start)
5. [How Docker enhances the experimental setup](#how-docker-enhances-the-experimental-setup)
6. [Multi-volunteer deployment](#multi-volunteer-deployment)
7. [Updating a running deployment](#updating-a-running-deployment)
8. [Image size comparison](#image-size-comparison)

---

## Why Docker?

| Without Docker | With Docker |
|---|---|
| `git pull` → `pip install -r requirements.txt` on every machine | `docker compose pull && docker compose up -d` |
| "It works on my machine" — Python 3.9 vs 3.12, missing system libs | Identical environment everywhere |
| 3–5 GB per machine (Ubuntu + Python + PyTorch + dependencies) | ~600 MB image, downloaded once |
| Manual process for N volunteer machines | One command per machine |
| Changing `compression.py` → full redeploy + reinstall | Volunteers download only the changed layer (a few KB) |

Docker turns the volunteer node into a **self-contained, reproducible artifact** that
runs identically on any Linux machine — lab workstations, cloud VMs, or donated hardware.

---

## Files overview

```
distributed_learning/
├── Dockerfile          # Image definition — how to build the volunteer container
├── .dockerignore       # Files excluded from the build context (smaller, faster builds)
├── docker-compose.yml  # Runtime configuration — env vars, volumes, networking
├── entrypoint.sh       # Container startup script — validates env vars, launches volunteer.py
└── requirements.txt    # Python dependencies (used by Dockerfile and bare-metal setup)
```

### Dockerfile

Defines the container image in ordered layers:

| Layer | Purpose | Change frequency |
|---|---|---|
| `FROM python:3.12-slim` | Lightweight Debian-based Python (~50 MB) | Never |
| `apt-get install build-essential` | C compiler for Python extensions (psutil) | Rarely |
| `COPY requirements.txt` + `pip install` | PyTorch, torchvision, numpy, psutil | When deps change |
| `COPY src/ volunteer.py` | Application source | Every code change |
| `USER volunteer` | Drop privileges to non-root | Never |

**Why the order matters:** changing `compression.py` only invalidates the last layers.
Volunteers pulling an update download a few KB instead of re-downloading the full
PyTorch stack (hundreds of MB).

### .dockerignore

Prevents 13 categories of unnecessary files from entering the build context:
virtual environments, `__pycache__`, IDE config, git history, results, logs, OS junk,
and generated `.json` / `.docx` / `.pdf` files.

**Impact:** smaller build context → faster `docker build` → smaller final image.

### docker-compose.yml

Describes how the container runs:

- **Environment variables** — 21 configurable parameters with sensible defaults,
  all sourced from a `.env` file. No experiment config is baked into the image.
- **Volume mount** — `./results:/app/results` persists experiment data outside the container.
- **Host networking** — `network_mode: host` lets the volunteer reach the coordinator
  and manager by their real IP. Essential for the XOR topology and gossip protocol.
- **Auto-restart** — `restart: unless-stopped` recovers from reboots without manual
  intervention.

### entrypoint.sh

The bridge between Docker environment variables and `volunteer.py` CLI arguments:

1. **Validates** that `VOLUNTEER_ID`, `COORDINATOR_HOST`, and `MANAGER_HOST` are set
2. **Builds** the argument list for `volunteer.py` from 21 optional env vars
3. **Exports** configuration variables with defaults (dataset, compression, gossip params, etc.)
4. **Creates** the results directory
5. **Launches** the volunteer with `exec python3 /app/volunteer.py`

### requirements.txt

Pinned Python dependencies for reproducible builds:
`torch`, `torchvision`, `numpy`, `psutil`.

CPU-only PyTorch is installed by default (~200 MB wheel). No CUDA, cuDNN, or NVIDIA
libraries are pulled — sufficient for MNIST/CIFAR-10 experiments.

---

## Design decisions

### 1. Slim base image avoids the 5 GB trap

A common anti-pattern in ML projects:

```dockerfile
# BAD — produces a 3–5 GB image
FROM ubuntu:noble
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch torchvision
```

Our approach:

```dockerfile
# GOOD — ~600 MB final image
FROM python:3.12-slim
RUN pip install --no-cache-dir torch torchvision
```

`python:3.12-slim` is Debian-based, ~50 MB, with Python pre-installed. The CPU-only
PyTorch wheel adds ~200 MB. The total image stays under 1 GB.

### 2. Experiment parameters never baked into the image

The same image runs every experiment. Configuration lives entirely in environment
variables:

```yaml
# docker-compose.yml
environment:
  - DATASET=${DATASET:-mnist}
  - COMPRESSION=${COMPRESSION:-quantization}
  - LOCAL_EPOCHS=${LOCAL_EPOCHS:-3}
```

To switch from MNIST/quantization to CIFAR-10/sparsification, change `.env` and
restart — no rebuild needed:

```bash
DATASET=cifar10 COMPRESSION=sparsification docker compose up -d
```

### 3. Layer caching for efficient updates

When a bug is fixed in `compression.py`, volunteers pull only the changed layer:

```
Layer 1 (python:3.12-slim)        → cached ✓ (50 MB — unchanged)
Layer 2 (build-essential)         → cached ✓ (unchanged)
Layer 3 (pip install)             → cached ✓ (requirements.txt unchanged)
Layer 4 (src/ + volunteer.py)    → NEW   (a few KB — this is the fix)
```

Compare this to the bare-metal workflow: `git pull` + `pip install -r requirements.txt`
on every machine, every time.

### 4. Non-root execution

The container runs as user `volunteer` (UID 1000), not root. This is standard Docker
security hardening — a compromised process inside the container has no root access
to the host.

**Note on volume permissions:** `./results` on the host must be writable by UID 1000.
If your host user has a different UID, adjust with:

```bash
chown -R 1000:1000 ./results
```

### 5. Python output unbuffered

`ENV PYTHONUNBUFFERED=1` ensures `print()` output and log lines appear immediately
in `docker compose logs -f`, rather than being held in a buffer until the process exits.

---

## Quick start

### Prerequisites

- Docker Engine ≥ 24.0 (or Docker Desktop)
- Docker Compose ≥ 2.20

### 1. Configure your environment

```bash
# Copy the example and edit it
cp .env.example .env

# Edit .env — set the IPs of your coordinator and manager
# COORDINATOR_HOST=192.168.1.10
# MANAGER_HOST=192.168.1.11
# VOLUNTEER_ID=0
```

### 2. Build and start

```bash
# Build the image and start the container in the background
docker compose up -d --build

# Follow logs
docker compose logs -f

# Stop
docker compose down
```

### 3. Verify it works

```bash
# Check the container is running
docker compose ps

# Check volunteer output
docker compose logs volunteer | head -30

# Check results are being written
ls -la ./results/
```

---

## How Docker enhances the experimental setup

### Reproducibility

Every volunteer runs the **exact same Python version, PyTorch version, and system
libraries**. No more debugging version mismatches across machines. A paper result
from 6 months ago can be reproduced by checking out the commit and running
`docker compose up`.

### Deployment speed

| Operation | Bare metal (5 machines) | Docker (5 machines) |
|---|---|---|
| Initial setup | ~15 min per machine (git, venv, pip) | `docker compose up -d` per machine |
| Code update | `git pull` + `pip install` on each | `docker compose pull && up -d` |
| Rollback | `git checkout <hash>` + reinstall deps | `image: <org>/volunteer:v1.2` then `up -d` |

### Experiment velocity

Because parameters are environment variables, not baked into the image, you can run
multiple experiment variants **in parallel on the same machine**:

```bash
# Terminal 1 — quantization experiment
VOLUNTEER_ID=0 COMPRESSION=quantization docker compose -p exp-quant up -d

# Terminal 2 — sparsification experiment (different project name = separate container)
VOLUNTEER_ID=1 COMPRESSION=sparsification docker compose -p exp-sparse up -d

# Terminal 3 — no compression (baseline)
VOLUNTEER_ID=2 COMPRESSION=none docker compose -p exp-none up -d
```

Each experiment writes to its own `results/` directory (configured via
`STATS_DIR`).

### Frugality

The image is ~600 MB — a fraction of the 3–5 GB a naive `ubuntu + python + pytorch`
build would produce. On a network of 20 volunteer machines, that's **~12 GB total
vs ~80–100 GB** of downloads. This respects the frugal computing ethos of the project.

---

## Multi-volunteer deployment

### On separate machines (production)

On each volunteer machine:

```bash
# Machine C (volunteer 0)
echo 'VOLUNTEER_ID=0
COORDINATOR_HOST=192.168.1.10
MANAGER_HOST=192.168.1.11' > .env
docker compose up -d

# Machine D (volunteer 1)
echo 'VOLUNTEER_ID=1
COORDINATOR_HOST=192.168.1.10
MANAGER_HOST=192.168.1.11' > .env
docker compose up -d
```

The coordinator and manager still run bare-metal (or in their own containers) —
Docker is only required on volunteer machines.

> **Why must `VOLUNTEER_ID` be set manually?** The coordinator already discovers
> volunteers dynamically by MAC address — it auto-adds new ones and auto-purges
> stale ones. However, the coordinator has no concept of `VOLUNTEER_ID`. That ID
> is used exclusively for **data partitioning** on the volunteer side: it
> determines which shard of the dataset this volunteer trains on (see
> [`src/dataset.py`](src/dataset.py)). If you skip `VOLUNTEER_ID`, every machine
> defaults to `0` (per [`docker-compose.yml`](docker-compose.yml#L30)),
> meaning **all volunteers would train on the same data shard** — they'd be
> redundant instead of complementary. The coordinator knows *that* N volunteers
> are connected, but it has no idea *which partition* each one trains on. For
> the coordinator to assign IDs dynamically, the protocol would need to be
> extended (volunteer requests an ID → coordinator assigns the next available →
> volunteer uses it for partitioning). Until then, assign a unique,
> zero-indexed `VOLUNTEER_ID` to each machine.

### On a single machine (testing)

Use distinct project names to isolate containers:

```bash
for id in 0 1 2 3 4; do
    VOLUNTEER_ID=$id \
    COORDINATOR_HOST=192.168.1.10 \
    MANAGER_HOST=192.168.1.11 \
    STATS_DIR=/app/results/volunteer_$id \
    docker compose -p volunteer$id up -d
done
```

---

## Updating a running deployment

When you fix a bug or change the source code:

```bash
# 1. Rebuild the image (only changed layers are rebuilt)
docker compose build

# 2. Restart with the new image
docker compose up -d

# If the image is pushed to a registry:
docker compose pull && docker compose up -d
```

Volunteers download only the layers that changed. A fix in `compression.py` results
in a ~5 KB pull, not a full 600 MB re-download.

---

## Image size comparison

| Approach | Base | Size (approx.) | Notes |
|---|---|---|---|
| Naive `ubuntu + python + torch` | ubuntu:noble | 3–5 GB | Includes full OS, CUDA libs, no layer optimization |
| This Dockerfile | python:3.12-slim | ~600 MB | CPU-only PyTorch, selective COPY, no bloat |
| With CUDA (not needed for MNIST) | nvidia/cuda:12.4-runtime + torch | 6–8 GB | Only needed for large models on GPU |

The 5–10× size reduction is achieved by:
- **Right base image:** `python:3.12-slim` instead of `ubuntu:noble`
- **CPU-only PyTorch:** no CUDA, cuDNN, or NVIDIA libraries
- **`.dockerignore`:** excludes venvs, caches, git history, generated files
- **`--no-install-recommends`:** skips optional apt packages

---

*Project — Apprentissage distribué frugal sur machines volontaires*
