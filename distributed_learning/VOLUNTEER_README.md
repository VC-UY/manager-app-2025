# Guide volontaire — nœud Docker léger

## 1. Contexte

Ce projet est conçu pour que des volontaires fassent tourner un nœud de calcul d'apprentissage distribué sur leur PC, sans rien installer directement sur le système.

Le nœud volontaire s'exécute uniquement dans un conteneur Docker. Cela signifie :
- pas de modification permanente du système d'exploitation,
- pas de programme Python installé globalement,
- tout est contenu dans Docker,
- les seuls fichiers persistants sont ceux que vous acceptez de monter sur votre machine.

### Ce qui est ajouté sur votre PC

Quand vous lancez le service volontaire avec `docker compose`, voici ce qui apparaît :
- un conteneur Docker `volunteer` qui tourne en arrière-plan,
- un volume monté depuis le répertoire `./results` du dossier `distributed_learning` dans le conteneur,
- le réseau Docker utilise le mode `host`, donc le conteneur partage la pile réseau de votre machine,
- une image Docker construite localement si l'image publique n'est pas disponible.

### Ce que vous verrez sur le disque

- `./results/` : dossier sur votre PC qui contient les fichiers de résultats/statistiques du nœud.
- `.env` : fichier de configuration contenant les `VOLUNTEER_ID`, `COORDINATOR_HOST`, `MANAGER_HOST`, etc.
- `docker` stocke aussi l'image et les couches localement : cela peut utiliser environ 0,5–1 Go d'espace disque.

> Note : le conteneur ne place pas de logs directement dans votre dossier, les logs sont fournis par Docker via `docker compose logs`.

## 2. Comment exécuter et arrêter

### Étape 1 — Préparer la configuration

Dans le dossier `distributed_learning` :

```bash
cp .env.example .env
```

Éditez ensuite `.env` et renseignez au minimum :
- `VOLUNTEER_ID` : un identifiant unique pour votre machine,
- `COORDINATOR_HOST` : l'adresse IP ou le nom d'hôte du coordinateur,
- `MANAGER_HOST` : l'adresse IP ou le nom d'hôte du manager.

### Étape 2 — Lancer le nœud volontaire

```bash
docker compose up -d
```

Le service se lance en arrière-plan. Si l'image publique est disponible, Docker la télécharge. Sinon, elle sera construite localement à partir du `Dockerfile`.

### Étape 3 — Vérifier que ça fonctionne

Voir l'état du conteneur :

```bash
docker compose ps
```

Suivre les logs en direct :

```bash
docker compose logs -f
```

### Étape 4 — Arrêter proprement

```bash
docker compose down
```

Cela arrête le nœud et supprime le réseau et les conteneurs générés par `docker compose`.

## 3. Comment désinstaller si on n'en veut plus

Si vous souhaitez supprimer toutes les traces de l'environnement du volontaire :

1. Arrêtez le conteneur si nécessaire :

```bash
./uninstall_volunteer.sh
```

2. Ce script fait automatiquement :
- `docker compose down --rmi all --volumes --remove-orphans`
- suppression du dossier `results/`
- suppression du fichier `.env`
- suppression des caches Python locaux `__pycache__`

### Commandes manuelles alternatives

Si vous préférez faire la désinstallation à la main :

```bash
docker compose down --rmi all --volumes --remove-orphans
rm -rf results .env
find . -type d -name '__pycache__' -exec rm -rf {} +
```

## 4. Ce que vous ne verrez pas

- pas de fichiers installés dans `/usr/bin` ou dans `C:\Program Files`,
- pas de modifications dans le système hôte hors du dossier du projet,
- pas de services système ajoutés en dehors du conteneur Docker.

### En résumé

Le rôle du volontaire est simple : fournir de la capacité de calcul via Docker. Vous gardez le contrôle total, vous pouvez arrêter le nœud à tout moment et tout supprimer proprement avec le script de désinstallation.
