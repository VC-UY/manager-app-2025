# Publishing the Volunteer Image to Docker Hub

> **Goal:** build the volunteer image once, push it to Docker Hub (free),
> and let every volunteer machine pull it with a single command — no local
> build, no Python/PyTorch install, no source clone.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Step 1 — Create a Docker Hub account](#step-1--create-a-docker-hub-account)
3. [Step 2 — Build the image](#step-2--build-the-image)
4. [Step 3 — Log in from the CLI](#step-3--log-in-from-the-cli)
5. [Step 4 — Push to Docker Hub](#step-4--push-to-docker-hub)
6. [Step 5 — Update docker-compose.yml](#step-5--update-docker-composeyml)
7. [Step 6 — Pull and run on a volunteer machine](#step-6--pull-and-run-on-a-volunteer-machine)
8. [Updating the image later](#updating-the-image-later)
9. [Version tags (recommended)](#version-tags-recommended)
10. [Free-tier limits to keep in mind](#free-tier-limits-to-keep-in-mind)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Docker Engine ≥ 24.0 installed on **your** machine (the one you'll build on)
- Docker Compose ≥ 2.20
- A web browser for account creation

---

## Step 1 — Create a Docker Hub account

1. Go to [hub.docker.com/signup](https://hub.docker.com/signup)
2. Pick a username (this will be your `<dockerhub-username>` everywhere)
3. Verify your email address

The free **Personal** plan gives you:
- Unlimited **public** repositories
- 1 private repository
- 100 image pulls per hour (per user)
- Enough for a research project with a handful of volunteer machines

---

## Step 2 — Build the image

Build and tag the image using your Docker Hub username. The tag must follow the
pattern `<username>/<image-name>:<version>` for Docker Hub to accept it.

```bash
# From the distributed_learning/ directory (where the Dockerfile lives)
docker build -t pablo93230/distributed-learning-volunteer:latest .
```

**Example** with a real username:

```bash
docker build -t jdoe/distributed-learning-volunteer:latest .
```

### What `-t` does

`-t` assigns a **tag** (name + version) to the image. The `:<version>` suffix
defaults to `:latest` if omitted. Using explicit versions (`:v1.0`, `:v1.1`)
alongside `:latest` is a good habit — see [Version tags](#version-tags-recommended).

### Build-time notes

| What | Why |
|---|---|
| `--no-cache` | Skip this unless you suspect a stale layer |
| First build | ~5 minutes (downloads `python:3.12-slim` + PyTorch) |
| Subsequent builds | Seconds to a minute (only changed layers rebuild) |

### Verify the build

```bash
docker images | grep distributed-learning-volunteer
# Expected output:
# jdoe/distributed-learning-volunteer   latest   abc123...   2 minutes ago   650MB
```

---

## Step 3 — Log in from the CLI

You only need to do this once per machine. Credentials are cached until you
explicitly `docker logout`.

```bash
docker login
```

You'll be prompted for your Docker Hub username and password. If you've enabled
2FA (recommended), generate a **Personal Access Token** at
[hub.docker.com/settings/security](https://hub.docker.com/settings/security) and
use that instead of your password.

### Alternative: log in without prompts

```bash
echo "<your-token>" | docker login -u <dockerhub-username> --password-stdin
```

This is useful in CI scripts, but **never** hardcode tokens in shell scripts
that are committed to git.

---

## Step 4 — Push to Docker Hub

```bash
docker push pablo93230/distributed-learning-volunteer:latest
```

The first push uploads every layer (~600 MB). Subsequent pushes upload only the
layers that changed — typically a few KB if you only modified Python source.

**Docker Hub creates the repository automatically** on first push. You don't
need to create it manually in the web UI.

### Verify on Docker Hub

Open `https://hub.docker.com/r/<dockerhub-username>/distributed-learning-volunteer`
in your browser. You should see the image with its tags.

Alternatively, check from the CLI:

```bash
docker search <dockerhub-username>/distributed-learning-volunteer
```

---

## Step 5 — Update `docker-compose.yml`

Edit the `image:` line in [`docker-compose.yml`](docker-compose.yml) to point to
your published image:

```yaml
services:
  volunteer:
    # Replace with your actual Docker Hub username
    image: <dockerhub-username>/distributed-learning-volunteer:latest

    # Keep the build: block as a fallback — if the image can't be pulled,
    # Docker Compose will build it locally.
    build:
      context: .
      dockerfile: Dockerfile

    # ... rest stays the same
```

> **Replace `<dockerhub-username>`** with your actual Docker Hub username.
> The placeholder `<your-dockerhub-username>` in the current
> [`docker-compose.yml`](docker-compose.yml) exists so you can do a
> project-wide find-and-replace.

### Why keep `build:` alongside `image:`?

When `image:` is present **and** available on Docker Hub, `docker compose up`
pulls it. But if a volunteer machine has no internet access, or you want to test
a local change without pushing, `docker compose build` still works — it builds
from the local Dockerfile.

---

## Step 6 — Pull and run on a volunteer machine

On every volunteer machine, you only need **two files**: `.env` and
`docker-compose.yml`. No Python, no PyTorch, no source code needed.

```bash
# 1. Copy only the necessary files to the volunteer machine
scp docker-compose.yml volunteer@machine-c:~/distributed_learning/

# 2. On the volunteer machine — set its unique ID and the coordinator/manager IPs
ssh volunteer@machine-c
cd ~/distributed_learning
echo 'VOLUNTEER_ID=0
COORDINATOR_HOST=192.168.1.10
MANAGER_HOST=192.168.1.11' > .env

# 3. Pull the image and start
docker compose pull    # downloads ~600 MB once
docker compose up -d   # starts the container in the background

# 4. Watch the logs
docker compose logs -f
```

**Repeat for each volunteer**, incrementing `VOLUNTEER_ID` every time (0, 1, 2,
…). See [`DOCKER.md` — Why must `VOLUNTEER_ID` be set manually?](DOCKER.md#on-separate-machines-production)
for the rationale.

---

## Updating the image later

When you fix a bug or change the source code:

```bash
# 1. Rebuild (only changed layers are rebuilt)
docker build -t <dockerhub-username>/distributed-learning-volunteer:latest .

# 2. Push (only changed layers are uploaded)
docker push <dockerhub-username>/distributed-learning-volunteer:latest
```

On each volunteer machine:

```bash
# Pull the updated layers (typically a few KB) and restart
docker compose pull && docker compose up -d
```

Thanks to Docker's layer caching, a fix in `compression.py` results in a
~5 KB download per volunteer, not a full 600 MB re-download.

### Auto-update with Watchtower

If you want volunteer machines to update automatically when you push a new
image, enable the optional Watchtower service in `docker-compose.yml`.

1. Start volunteer + Watchtower once:

```bash
docker compose up -d
```

2. Push a new image from your build machine:

```bash
docker build -t <dockerhub-username>/distributed-learning-volunteer:latest .
docker push <dockerhub-username>/distributed-learning-volunteer:latest
```

3. Watchtower checks Docker Hub every 5 minutes (`--interval 300`) and will
pull the new image + restart the volunteer container automatically.

You can also use the helper script `enable_watchtower.sh` in the project root
on the volunteer machine to start both services in one command.

> Note: Watchtower requires access to the host Docker socket
> (`/var/run/docker.sock`). This means the volunteer host must run Docker
> locally and allow the socket mount.

---

## Version tags (recommended)

`latest` is convenient during development, but for experiment reproducibility,
tag each release:

```bash
# Tag the same image with both :latest and a version
docker build -t <dockerhub-username>/distributed-learning-volunteer:v1.0 .
docker tag <dockerhub-username>/distributed-learning-volunteer:v1.0 \
           <dockerhub-username>/distributed-learning-volunteer:latest

# Push both tags
docker push <dockerhub-username>/distributed-learning-volunteer:v1.0
docker push <dockerhub-username>/distributed-learning-volunteer:latest
```

Then, to reproduce the exact environment from a paper:

```bash
# In docker-compose.yml, pin to a specific version
image: <dockerhub-username>/distributed-learning-volunteer:v1.0
```

This is the Docker equivalent of pinning Python dependencies — you can always
go back to the exact byte-for-byte image that produced a given result.

**Suggested versioning scheme:**

| Tag | Meaning |
|---|---|
| `:v1.0` | First stable release |
| `:v1.1` | Bug fix, minor improvement |
| `:v2.0` | Breaking change (e.g., new protocol) |
| `:latest` | Always the most recent push (for dev) |

---

## Free-tier limits to keep in mind

From [docker.com/pricing](https://www.docker.com/pricing/) (Personal plan):

| Resource | Limit | Is it a problem? |
|---|---|---|
| Public repos | Unlimited | No |
| Private repos | 1 | No (keep the volunteer image public) |
| Pulls per hour | 100 per user | No — even 20 volunteers pulling once an hour stay under |
| Storage | Unlimited (fair use) | No — a 600 MB image is tiny by Docker Hub standards |
| Users | 1 | No (only you publish) |

The 100 pulls/hour limit is per **authenticated user**, not per image.
If each volunteer pulls the image once at startup and you have fewer than
100 machines, you'll never hit it. Even if you do, the limit resets each hour.

### If you want to keep the image private

The free tier allows **1 private repository**. To use it:

```bash
# Do NOT put the image name in a public docker-compose.yml.
# Instead, have volunteers authenticate, then pull:
ssh volunteer@machine-c
docker login
docker pull <dockerhub-username>/distributed-learning-volunteer:latest
```

But for a research project, a public image is simpler — the code is already
open-source, and a public image means volunteers don't need Docker Hub accounts.

---

## Troubleshooting

### `denied: requested access to the resource is denied`

You're not logged in, or your token has expired.

```bash
docker logout && docker login
docker push <dockerhub-username>/distributed-learning-volunteer:latest
```

### `repository does not exist or may require 'docker login'`

Double-check the tag matches your Docker Hub username **exactly**
(case-sensitive). The image name must be `<username>/<repo-name>:<tag>`.

### `unauthorized: authentication required` on a volunteer machine

The volunteer image is private, and the volunteer machine isn't authenticated.
Either make the image **public** (Settings → Visibility on Docker Hub) or run
`docker login` on each volunteer machine.

### Push is very slow on a poor connection

Docker pushes layers in parallel only within a single `docker push`. On a slow
connection, consider breaking the push into chunks:

```bash
# Push each layer sequentially (slower parallelism, but more resilient)
docker push --quiet <dockerhub-username>/distributed-learning-volunteer:latest
```

If the push times out, it's safe to retry — Docker resumes from where it
stopped.

### `buildx` vs legacy builder

If you see warnings about `buildx`:

```bash
# Use the modern builder (multi-platform compatible, faster)
docker buildx build --tag <dockerhub-username>/distributed-learning-volunteer:latest .
```

For CPU-only PyTorch on x86-64 Linux, the legacy builder works fine. `buildx`
matters if you ever want ARM64 volunteers (e.g., Raspberry Pi).

---

## Quick reference card

```bash
# ─── First time ──────────────────────────────────────────────────────────
docker login                                                       # once per machine
docker build -t <you>/distributed-learning-volunteer:latest .      # build
docker push <you>/distributed-learning-volunteer:latest            # publish

# ─── After a code change ────────────────────────────────────────────────
docker build -t <you>/distributed-learning-volunteer:latest .      # rebuild (fast)
docker push <you>/distributed-learning-volunteer:latest            # push delta

# ─── On each volunteer machine ──────────────────────────────────────────
echo 'VOLUNTEER_ID=<N>
COORDINATOR_HOST=<ip>
MANAGER_HOST=<ip>' > .env
docker compose pull && docker compose up -d
```

---

*Project — Apprentissage distribué frugal sur machines volontaires*
