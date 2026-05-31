# Docker-Based Deployment Strategy for the Distributed Volunteer Computing Platform

## 1. Introduction

The current distributed computing platform implements a volunteer-based federated learning system composed of three main actors:

* A **Coordinator**, responsible for maintaining volunteer information and network topology.
* A **Manager**, responsible for orchestration and global supervision.
* Multiple **Volunteer Nodes**, responsible for performing local computation and exchanging model updates.

The platform relies on a Python environment containing several scientific and machine-learning dependencies, including PyTorch, Torchvision, NumPy, and associated networking libraries.

While the software executes correctly in a controlled development environment, deployment on volunteer machines introduces several practical challenges:

* Different Linux distributions and package versions.
* Dependency installation failures.
* Slow or unstable Internet connections.
* Difficulty reproducing identical execution environments.
* Increased support effort when volunteers encounter configuration issues.

To address these challenges, Docker is introduced as the primary deployment mechanism for volunteer nodes.

---

# 2. Motivation for Docker Adoption

## 2.1 Environment Reproducibility

The volunteer computing experiment requires all participants to execute the same code under identical software conditions.

Without containerization, volunteers would need to manually install:

* Python
* PyTorch
* Torchvision
* NumPy
* Scientific libraries
* Networking dependencies

Differences in operating system versions, package repositories, and Python environments may lead to inconsistent behavior.

Docker solves this problem by packaging:

* The operating system layer
* Python runtime
* All required libraries
* The application source code

into a single immutable image.

Every volunteer therefore executes the exact same software stack.

---

## 2.2 Simplified Volunteer Onboarding

The objective is to minimize technical requirements for volunteers.

Instead of requiring:

```bash
git clone ...
pip install -r requirements.txt
python volunteer.py
```

a volunteer only needs:

```bash
docker compose up -d
```

This significantly reduces setup complexity and lowers the barrier to participation.

---

## 2.3 Reduced Dependency Downloads

Machine-learning frameworks such as PyTorch are large packages.

Repeated installation of dependencies is undesirable, especially in environments where Internet access is unstable.

Docker images allow all dependencies to be downloaded once and reused indefinitely.

After the initial image pull, volunteers can restart containers without re-downloading any libraries.

---

## 2.4 Platform Independence

Although the project currently targets Linux systems, volunteers may use:

* Ubuntu
* Debian
* Fedora
* Arch Linux
* Other distributions

Docker provides a consistent runtime environment across all supported Linux hosts.

This removes distribution-specific deployment issues.

---

## 2.5 Experiment Reproducibility

A scientific experiment should be reproducible.

Using a versioned Docker image ensures that:

* Every participant executes the same code.
* The same dependency versions are used.
* Experimental results can be reproduced later.

This is particularly important when evaluating distributed-learning performance.

---

# 3. Design Decision: Application Code Inside the Image

Two deployment strategies were considered.

## Option A: Mount Application Source Code

The container only contains dependencies.

Application code is provided through a mounted volume.

Example:

```yaml
volumes:
  - ./src:/app/src
```

Advantages:

* Easy development.
* Source code can be modified without rebuilding images.

Disadvantages:

* Volunteers must maintain a Git repository.
* Volunteers must execute updates manually.
* Version mismatches may occur.
* Increased support complexity.

---

## Option B: Package Source Code Inside the Image

The Docker image contains:

* Application source code
* Python runtime
* Dependencies
* Datasets

Advantages:

* Single-command deployment.
* No Git installation required.
* No source synchronization issues.
* Maximum reproducibility.

Disadvantages:

* Code changes require image updates.

---

## Selected Approach

The project adopts **Option B**.

The volunteer image will contain the complete application.

This approach prioritizes simplicity, reproducibility, and ease of participation.

---

# 4. Docker Architecture

The deployment architecture consists of two distinct environments.

## Development Environment

Used by developers.

Characteristics:

* Source code mounted from host.
* Fast iteration.
* Frequent code changes.

Purpose:

* Testing.
* Debugging.
* Experiment development.

---

## Volunteer Environment

Used by external participants.

Characteristics:

* Fully packaged image.
* No mounted source code.
* Minimal setup requirements.

Purpose:

* Stable experiment execution.

---

# 5. Docker Components

## 5.1 Dockerfile

The Dockerfile defines the volunteer execution environment.

Responsibilities:

### Base Operating System

A lightweight Linux image will be selected.

Example:

```dockerfile
FROM python:3.12-slim
```

This avoids unnecessary packages and reduces image size.

---

### Dependency Installation

The Dockerfile installs:

* PyTorch (CPU version)
* Torchvision
* NumPy
* Other Python dependencies

GPU support is intentionally excluded.

The current experiments target CPU execution only.

Removing CUDA-related packages significantly reduces image size.

---

### Dataset Integration

The Docker image may optionally include:

* MNIST dataset
* Other experimental datasets

Benefits:

* No runtime downloads.
* Better operation in unstable network environments.
* Faster startup times.

---

### Application Code

The Dockerfile copies:

```text
src/
volunteer.py
entrypoint.sh
```

into the container.

This creates a self-contained execution environment.

---

### Startup Configuration

The Dockerfile specifies the startup script:

```dockerfile
ENTRYPOINT ["./entrypoint.sh"]
```

---

## 5.2 .dockerignore

The .dockerignore file reduces build size.

It prevents unnecessary files from entering the image.

Typical exclusions:

```text
.git
results/
__pycache__/
*.pyc
```

Benefits:

* Faster builds.
* Smaller images.
* Reduced upload size.

---

## 5.3 entrypoint.sh

The entrypoint script acts as the container startup controller.

Responsibilities:

### Environment Validation

Verify required variables:

```text
COORDINATOR_HOST
MANAGER_HOST
```

---

### Runtime Preparation

Create:

```text
results/
logs/
```

directories if needed.

---

### Launch Volunteer Process

Execute:

```bash
python volunteer.py
```

---

### Graceful Shutdown

Handle container stop signals.

Allow volunteer statistics to be saved before termination.

---

## 5.4 docker-compose.yml

This is the only file most volunteers interact with.

Responsibilities:

### Container Deployment

Defines the volunteer service.

---

### Environment Variables

Example:

```yaml
environment:
  COORDINATOR_HOST: x.x.x.x
  MANAGER_HOST: x.x.x.x
```

---

### Restart Policy

Example:

```yaml
restart: unless-stopped
```

This ensures automatic recovery after reboots.

---

### Persistent Storage

Optional volume mappings:

```yaml
volumes:
  - volunteer_results:/app/results
```

This preserves statistics between container restarts.

---

# 6. Update Strategy

Docker images will be versioned.

Example:

```text
volunteer:v1
volunteer:v2
volunteer:v3
```

When application code changes:

1. A new image is built.
2. The image is pushed to a registry.
3. Volunteers execute:

```bash
docker compose pull
docker compose up -d
```

Docker downloads only modified image layers.

Consequently, volunteers do not repeatedly download:

* Python
* PyTorch
* Torchvision

Only updated application layers are transferred.

This minimizes bandwidth consumption while preserving reproducibility.

---

# 7. Expected Benefits

The proposed Docker-based architecture provides:

* Identical execution environments for all volunteers.
* Simplified deployment procedures.
* Reduced dependency-management burden.
* Improved experiment reproducibility.
* Reduced support requirements.
* Efficient update distribution.
* Better operation in low-bandwidth environments.

These benefits directly support the objectives of the distributed volunteer computing experiment and justify the adoption of Docker as the primary deployment mechanism.

---

# 8. Conclusion

Docker provides a practical solution for deploying volunteer nodes in the distributed federated-learning platform.

By packaging the operating system, dependencies, datasets, and application code into a single image, the project eliminates environment inconsistencies while significantly simplifying volunteer participation.

The selected design favors a fully self-contained volunteer image, distributed through a container registry and executed through Docker Compose. This approach maximizes reproducibility, minimizes setup effort, and creates a stable foundation for future experimentation.
