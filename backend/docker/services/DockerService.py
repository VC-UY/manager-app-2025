import os
import json
import logging
import tempfile
import subprocess
import shutil
from django.conf import settings
from tasks.models import Task, TaskStatus

logger = logging.getLogger(__name__)

class DockerService:
    """
    Service pour la gestion des images Docker pour les workflows matriciels
    """
    
    def __init__(self):
        """
        Initialise le service Docker avec les paramètres de configuration
        """
        self.registry = getattr(settings, 'DOCKER_REGISTRY', 'localhost:5000')
        self.namespace = getattr(settings, 'DOCKER_NAMESPACE', 'matrixflow')
        self.push_enabled = getattr(settings, 'DOCKER_PUSH_ENABLED', True)
        self.docker_templates_dir = getattr(settings, 'DOCKER_TEMPLATES_DIR', 
                                          os.path.join(settings.BASE_DIR, 'docker', 'templates'))
        self.base_image = getattr(settings, 'DOCKER_BASE_IMAGE', 'python:3.9-slim')
        
        # S'assurer que le répertoire des templates existe
        os.makedirs(self.docker_templates_dir, exist_ok=True)
    
    def build_image_for_task(self, task):
        """
        Construit une image Docker pour une tâche spécifique
        
        Args:
            task (Task): La tâche pour laquelle construire l'image
            
        Returns:
            str: Nom complet de l'image Docker construite
        
        Raises:
            ValueError: Si le type de tâche n'est pas supporté ou en cas d'erreur
        """
        logger.info(f"Construction de l'image Docker pour la tâche {task.id}")
        
        # Déterminer le type de tâche
        command = task.command
        
        if command.startswith("matrix_add"):
            return self._build_matrix_add_image(task)
        elif command.startswith("matrix_multiply"):
            return self._build_matrix_multiply_image(task)
        else:
            raise ValueError(f"Type de tâche non supporté: {command}")
    
    def _build_matrix_add_image(self, task):
        """
        Construit une image Docker pour l'addition matricielle
        
        Args:
            task (Task): La tâche d'addition matricielle
            
        Returns:
            str: Nom complet de l'image Docker construite
        """
        image_name = f"{self.namespace}/matrix_addition"
        image_tag = f"{task.id}"
        image_full_name = f"{self.registry}/{image_name}:{image_tag}"
        
        # Créer un répertoire temporaire pour la construction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer le Dockerfile
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, 'w') as f:
                f.write(f"""
FROM {self.base_image}

# Installation des dépendances
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Installation des bibliothèques Python
RUN pip install --no-cache-dir numpy==1.23.5 requests==2.28.1 paho-mqtt==1.6.1

# Créer les répertoires nécessaires
RUN mkdir -p /app/data /app/results

# Copie des scripts
COPY worker.py /app/
COPY matrix_operations/ /app/matrix_operations/
COPY utils/ /app/utils/

# Variables d'environnement
ENV BROKER_URL="mqtt://broker.local:1883" \\
    TASK_ID="" \\
    VOLUNTEER_ID="" \\
    TASK_TYPE="matrix_addition" \\
    MATRIX_BLOCK_ROW_START=-1 \\
    MATRIX_BLOCK_ROW_END=-1 \\
    MATRIX_BLOCK_COL_START=-1 \\
    MATRIX_BLOCK_COL_END=-1

# Point d'entrée
WORKDIR /app
ENTRYPOINT ["python", "worker.py"]
                """)
            
            # Créer le script worker.py
            with open(os.path.join(temp_dir, "worker.py"), 'w') as f:
                f.write("""#!/usr/bin/env python3
                    import os
                    import sys
                    import json
                    import time
                    import logging
                    import paho.mqtt.client as mqtt
                    import requests
                    import numpy as np
                    from matrix_operations.addition import add_matrices_block
                    from utils.messaging import publish_status, publish_results

                    # Configuration du logging
                    logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    logger = logging.getLogger('worker')

                    # Récupération des variables d'environnement
                    broker_url = os.environ.get('BROKER_URL', 'mqtt://broker.local:1883')
                    task_id = os.environ.get('TASK_ID', '')
                    volunteer_id = os.environ.get('VOLUNTEER_ID', '')
                    task_type = os.environ.get('TASK_TYPE', 'matrix_addition')

                    # Décomposer l'URL du broker
                    broker_parts = broker_url.replace('mqtt://', '').split(':')
                    broker_host = broker_parts[0]
                    broker_port = int(broker_parts[1]) if len(broker_parts) > 1 else 1883

                    # Coordonnées du bloc matriciel
                    block_row_start = int(os.environ.get('MATRIX_BLOCK_ROW_START', -1))
                    block_row_end = int(os.environ.get('MATRIX_BLOCK_ROW_END', -1))
                    block_col_start = int(os.environ.get('MATRIX_BLOCK_COL_START', -1))
                    block_col_end = int(os.environ.get('MATRIX_BLOCK_COL_END', -1))

                    def download_matrices(config):
=                        matrix_a_source = config.get('matrix_a_source', '')
                        matrix_b_source = config.get('matrix_b_source', '')
                        
                        if not matrix_a_source or not matrix_b_source:
                            raise ValueError("Sources des matrices non spécifiées")
                        
                        # Télécharger la matrice A
                        logger.info(f"Téléchargement de la matrice A depuis {matrix_a_source}")
                        response_a = requests.get(matrix_a_source, timeout=30)
                        response_a.raise_for_status()
                        
                        # Télécharger la matrice B
                        logger.info(f"Téléchargement de la matrice B depuis {matrix_b_source}")
                        response_b = requests.get(matrix_b_source, timeout=30)
                        response_b.raise_for_status()
                        
                        # Extraire les blocs nécessaires
                        matrix_a = np.array(json.loads(response_a.text))
                        matrix_b = np.array(json.loads(response_b.text))
                        
                        # Vérifier que les matrices ont les mêmes dimensions
                        if matrix_a.shape != matrix_b.shape:
                            raise ValueError(f"Les matrices n'ont pas les mêmes dimensions: A={matrix_a.shape}, B={matrix_b.shape}")
                        
                        # Extraire le bloc si des coordonnées sont spécifiées
                        if block_row_start >= 0 and block_row_end >= 0 and block_col_start >= 0 and block_col_end >= 0:
                            matrix_a_block = matrix_a[block_row_start:block_row_end+1, block_col_start:block_col_end+1]
                            matrix_b_block = matrix_b[block_row_start:block_row_end+1, block_col_start:block_col_end+1]
                            return matrix_a_block, matrix_b_block
                        
                        return matrix_a, matrix_b

                    def process_matrix_addition(config):
                        try:
                            # Télécharger les matrices
                            matrix_a, matrix_b = download_matrices(config)
                            
                            # Reporter la progression initiale
                            publish_status(broker_host, broker_port, task_id, volunteer_id, 10)
                            
                            # Effectuer l'addition
                            logger.info(f"Début de l'addition matricielle pour le bloc: [{block_row_start}:{block_row_end}, {block_col_start}:{block_col_end}]")
                            result = add_matrices_block(matrix_a, matrix_b)
                            
                            # Reporter la progression
                            publish_status(broker_host, broker_port, task_id, volunteer_id, 80)
                            
                            # Préparer les résultats
                            results = {
                                'block_data': result.tolist(),
                                'dimensions': result.shape,
                                'position': {
                                    'row_start': block_row_start,
                                    'row_end': block_row_end,
                                    'col_start': block_col_start,
                                    'col_end': block_col_end
                                },
                                'timestamp': time.time()
                            }
                            
                            # Enregistrer les résultats localement
                            with open('/app/results/addition_result.json', 'w') as f:
                                json.dump(results, f)
                            
                            # Publier les résultats
                            logger.info("Publication des résultats de l'addition matricielle")
                            publish_results(broker_host, broker_port, task_id, volunteer_id, results)
                            
                            # Reporter la progression finale
                            publish_status(broker_host, broker_port, task_id, volunteer_id, 100)
                            
                            return True
                        
                        except Exception as e:
                            logger.error(f"Erreur lors du traitement de l'addition matricielle: {e}")
                            # Publier l'erreur
                            error_details = {
                                'error': str(e),
                                'timestamp': time.time()
                            }
                            publish_results(broker_host, broker_port, task_id, volunteer_id, None, error_details)
                            return False

                    def main():
                        if not task_id or not volunteer_id:
                            logger.error("Identifiants de tâche ou de volontaire non spécifiés")
                            sys.exit(1)
                        
                        logger.info(f"Démarrage du worker pour la tâche {task_id} par le volontaire {volunteer_id}")
                        
                        # Récupérer la configuration de la tâche si besoin
                        # Dans ce cas, la configuration est passée via les variables d'environnement
                        
                        config = {
                            'matrix_a_source': os.environ.get('MATRIX_A_SOURCE', ''),
                            'matrix_b_source': os.environ.get('MATRIX_B_SOURCE', ''),
                            'output_format': os.environ.get('OUTPUT_FORMAT', 'json')
                        }
                        
                        # Traiter la tâche selon son type
                        if task_type == 'matrix_addition':
                            success = process_matrix_addition(config)
                        else:
                            logger.error(f"Type de tâche non supporté: {task_type}")
                            sys.exit(1)
                        
                        if success:
                            logger.info("Traitement terminé avec succès")
                            sys.exit(0)
                        else:
                            logger.error("Traitement terminé avec erreur")
                            sys.exit(1)

                    if __name__ == "__main__":
                        main()
                """)
            
            # Créer le module matrix_operations
            os.makedirs(os.path.join(temp_dir, "matrix_operations"), exist_ok=True)
            with open(os.path.join(temp_dir, "matrix_operations", "__init__.py"), 'w') as f:
                f.write("")
            
            with open(os.path.join(temp_dir, "matrix_operations", "addition.py"), 'w') as f:
                f.write("""
                            import numpy as np
                            import logging

                            logger = logging.getLogger(__name__)

                            def add_matrices_block(matrix_a_block, matrix_b_block):

                                logger.info(f"Addition de blocs de matrices de dimensions {matrix_a_block.shape}")
                                
                                # Vérifier que les matrices ont les mêmes dimensions
                                if matrix_a_block.shape != matrix_b_block.shape:
                                    raise ValueError(f"Les blocs n'ont pas les mêmes dimensions: A={matrix_a_block.shape}, B={matrix_b_block.shape}")
                                
                                # Effectuer l'addition
                                result = matrix_a_block + matrix_b_block
                                
                                return result
                        """)
            
            # Créer le module utils
            os.makedirs(os.path.join(temp_dir, "utils"), exist_ok=True)
            with open(os.path.join(temp_dir, "utils", "__init__.py"), 'w') as f:
                f.write("")
            
            with open(os.path.join(temp_dir, "utils", "messaging.py"), 'w') as f:
                f.write("""
                            import json
                            import logging
                            import paho.mqtt.client as mqtt

                            logger = logging.getLogger(__name__)

                            def publish_status(broker_host, broker_port, task_id, volunteer_id, progress):

                                try:
                                    client = mqtt.Client(f"worker-{volunteer_id}-{task_id}")
                                    client.connect(broker_host, broker_port, 60)
                                    
                                    # Préparer le message
                                    message = {
                                        "task_id": task_id,
                                        "volunteer_id": volunteer_id,
                                        "progress": progress,
                                        "status": "running" if progress < 100 else "completed"
                                    }
                                    
                                    # Publier le message
                                    result = client.publish("tasks/status", json.dumps(message), qos=1)
                                    
                                    # Attendre que le message soit envoyé
                                    result.wait_for_publish()
                                    
                                    # Déconnecter le client
                                    client.disconnect()
                                    
                                    logger.info(f"Statut publié: {progress}% pour la tâche {task_id}")
                                    
                                    return True
                                
                                except Exception as e:
                                    logger.error(f"Erreur lors de la publication du statut: {e}")
                                    return False

                            def publish_results(broker_host, broker_port, task_id, volunteer_id, results, error_details=None):
                            
                                try:
                                    client = mqtt.Client(f"worker-{volunteer_id}-{task_id}")
                                    client.connect(broker_host, broker_port, 60)
                                    
                                    # Préparer le message
                                    message = {
                                        "task_id": task_id,
                                        "volunteer_id": volunteer_id,
                                        "status": "completed" if error_details is None else "failed",
                                        "results_location": {
                                            "protocol": "http",
                                            "host": "{{WORKER_IP}}",  # Sera remplacé par l'IP du volontaire
                                            "port": 8080,
                                            "basePath": "/results",
                                            "files": ["/addition_result.json"]
                                        } if results else None,
                                        "error_details": error_details
                                    }
                                    
                                    # Publier le message
                                    result = client.publish("tasks/results", json.dumps(message), qos=1)
                                    
                                    # Attendre que le message soit envoyé
                                    result.wait_for_publish()
                                    
                                    # Déconnecter le client
                                    client.disconnect()
                                    
                                    logger.info(f"Résultats publiés pour la tâche {task_id}")
                                    
                                    return True
                                
                                except Exception as e:
                                    logger.error(f"Erreur lors de la publication des résultats: {e}")
                                    return False
                        """)
            
            # Construction de l'image
            try:
                subprocess.run(
                    ["docker", "build", "-t", image_full_name, "."],
                    cwd=temp_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                logger.info(f"Image {image_full_name} construite avec succès")
                
                # Pousser l'image vers le registre Docker si activé
                if self.push_enabled:
                    self._push_image(image_full_name)
                
                return image_full_name
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Erreur lors de la construction de l'image: {e}")
                logger.error(f"STDOUT: {e.stdout.decode('utf-8')}")
                logger.error(f"STDERR: {e.stderr.decode('utf-8')}")
                raise ValueError(f"Erreur lors de la construction de l'image: {e}")
    
    def _build_matrix_multiply_image(self, task):
        """
        Construit une image Docker pour la multiplication matricielle
        
        Args:
            task (Task): La tâche de multiplication matricielle
            
        Returns:
            str: Nom complet de l'image Docker construite
        """
        image_name = f"{self.namespace}/matrix_multiplication"
        image_tag = f"{task.id}"
        image_full_name = f"{self.registry}/{image_name}:{image_tag}"
        
        # Créer un répertoire temporaire pour la construction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer le Dockerfile
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, 'w') as f:
                f.write(f"""
FROM {self.base_image}

# Installation des dépendances
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Installation des bibliothèques Python
RUN pip install --no-cache-dir numpy==1.23.5 requests==2.28.1 paho-mqtt==1.6.1

# Créer les répertoires nécessaires
RUN mkdir -p /app/data /app/results

# Copie des scripts
COPY worker.py /app/
COPY matrix_operations/ /app/matrix_operations/
COPY utils/ /app/utils/

# Variables d'environnement
ENV BROKER_URL="mqtt://broker.local:1883" \\
    TASK_ID="" \\
    VOLUNTEER_ID="" \\
    TASK_TYPE="matrix_multiplication" \\
    MATRIX_BLOCK_ROW_START=-1 \\
    MATRIX_BLOCK_ROW_END=-1 \\
    MATRIX_BLOCK_COL_START=-1 \\
    MATRIX_BLOCK_COL_END=-1 \\
    ALGORITHM="standard"

# Point d'entrée
WORKDIR /app
ENTRYPOINT ["python", "worker.py"]
                """)
            
            # Créer le script worker.py
            with open(os.path.join(temp_dir, "worker.py"), 'w') as f:
                f.write("""#!/usr/bin/env python3
                    import os
                    import sys
                    import json
                    import time
                    import logging
                    import paho.mqtt.client as mqtt
                    import requests
                    import numpy as np
                    from matrix_operations.multiplication import multiply_matrices_block
                    from utils.messaging import publish_status, publish_results

                    # Configuration du logging
                    logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    logger = logging.getLogger('worker')

                    # Récupération des variables d'environnement
                    broker_url = os.environ.get('BROKER_URL', 'mqtt://broker.local:1883')
                    task_id = os.environ.get('TASK_ID', '')
                    volunteer_id = os.environ.get('VOLUNTEER_ID', '')
                    task_type = os.environ.get('TASK_TYPE', 'matrix_multiplication')
                    algorithm = os.environ.get('ALGORITHM', 'standard')

                    # Décomposer l'URL du broker
                    broker_parts = broker_url.replace('mqtt://', '').split(':')
                    broker_host = broker_parts[0]
                    broker_port = int(broker_parts[1]) if len(broker_parts) > 1 else 1883

                    # Coordonnées du bloc matriciel
                    block_row_start = int(os.environ.get('MATRIX_BLOCK_ROW_START', -1))
                    block_row_end = int(os.environ.get('MATRIX_BLOCK_ROW_END', -1))
                    block_col_start = int(os.environ.get('MATRIX_BLOCK_COL_START', -1))
                    block_col_end = int(os.environ.get('MATRIX_BLOCK_COL_END', -1))

                    def download_matrices(config):
                        matrix_a_source = config.get('matrix_a_source', '')
                        matrix_b_source = config.get('matrix_b_source', '')
                        
                        if not matrix_a_source or not matrix_b_source:
                            raise ValueError("Sources des matrices non spécifiées")
                        
                        # Télécharger la matrice A
                        logger.info(f"Téléchargement de la matrice A depuis {matrix_a_source}")
                        response_a = requests.get(matrix_a_source, timeout=30)
                        response_a.raise_for_status()
                        
                        # Télécharger la matrice B
                        logger.info(f"Téléchargement de la matrice B depuis {matrix_b_source}")
                        response_b = requests.get(matrix_b_source, timeout=30)
                        response_b.raise_for_status()
                        
                        # Extraire les matrices complètes
                        matrix_a = np.array(json.loads(response_a.text))
                        matrix_b = np.array(json.loads(response_b.text))
                        
                        # Vérifier que les matrices sont compatibles pour la multiplication
                        if matrix_a.shape[1] != matrix_b.shape[0]:
                            raise ValueError(f"Les matrices ne sont pas compatibles pour la multiplication: "
                                            f"A={matrix_a.shape}, B={matrix_b.shape}")
                        
                        # Pour la multiplication, on a besoin de lignes complètes de A et de colonnes complètes de B
                        if block_row_start >= 0 and block_row_end >= 0 and block_col_start >= 0 and block_col_end >= 0:
                            # Extraire les lignes de A nécessaires pour ce bloc
                            matrix_a_rows = matrix_a[block_row_start:block_row_end+1, :]
                            
                            # Extraire les colonnes de B nécessaires pour ce bloc
                            matrix_b_cols = matrix_b[:, block_col_start:block_col_end+1]
                            
                            return matrix_a_rows, matrix_b_cols, (block_row_end - block_row_start + 1, block_col_end - block_col_start + 1)
                        
                        # Si pas de bloc spécifié, retourner les matrices complètes
                        return matrix_a, matrix_b, None

                    def process_matrix_multiplication(config):
                        try:
                            # Télécharger les matrices
                            matrix_a, matrix_b, result_shape = download_matrices(config)
                            
                            # Reporter la progression initiale
                            publish_status(broker_host, broker_port, task_id, volunteer_id, 10)
                            
                            # Effectuer la multiplication
                            logger.info(f"Début de la multiplication matricielle pour le bloc: [{block_row_start}:{block_row_end}, {block_col_start}:{block_col_end}], algorithme: {algorithm}")
                            result = multiply_matrices_block(matrix_a, matrix_b, algorithm, result_shape)
                            
                            # Reporter la progression
                            publish_status(broker_host, broker_port, task_id, volunteer_id, 80)
                            
                            # Préparer les résultats
                            results = {
                                'block_data': result.tolist(),
                                'dimensions': result.shape,
                                'position': {
                                    'row_start': block_row_start,
                                    'row_end': block_row_end,
                                    'col_start': block_col_start,
                                    'col_end': block_col_end
                                },
                                'algorithm': algorithm,
                                'timestamp': time.time()
                            }
                            
                            # Enregistrer les résultats localement
                            with open('/app/results/multiplication_result.json', 'w') as f:
                                json.dump(results, f)
                            
                            # Publier les résultats
                            logger.info("Publication des résultats de la multiplication matricielle")
                            publish_results(broker_host, broker_port, task_id, volunteer_id, results)
                            
                            # Reporter la progression finale
                            publish_status(broker_host, broker_port, task_id, volunteer_id, 100)
                            
                            return True
                        
                        except Exception as e:
                            logger.error(f"Erreur lors du traitement de la multiplication matricielle: {e}")
                            # Publier l'erreur
                            error_details = {
                                'error': str(e),
                                'timestamp': time.time()
                            }
                            publish_results(broker_host, broker_port, task_id, volunteer_id, None, error_details)
                            return False

                    def main():
                        if not task_id or not volunteer_id:
                            logger.error("Identifiants de tâche ou de volontaire non spécifiés")
                            sys.exit(1)
                        
                        logger.info(f"Démarrage du worker pour la tâche {task_id} par le volontaire {volunteer_id}")
                        
                        # Récupérer la configuration de la tâche si besoin
                        # Dans ce cas, la configuration est passée via les variables d'environnement
                        
                        config = {
                            'matrix_a_source': os.environ.get('MATRIX_A_SOURCE', ''),
                            'matrix_b_source': os.environ.get('MATRIX_B_SOURCE', ''),
                            'algorithm': algorithm,
                            'output_format': os.environ.get('OUTPUT_FORMAT', 'json')
                        }
                        
                        # Traiter la tâche selon son type
                        if task_type == 'matrix_multiplication':
                            success = process_matrix_multiplication(config)
                        else:
                            logger.error(f"Type de tâche non supporté: {task_type}")
                            sys.exit(1)
                        
                        if success:
                            logger.info("Traitement terminé avec succès")
                            sys.exit(0)
                        else:
                            logger.error("Traitement terminé avec erreur")
                            sys.exit(1)

                    if __name__ == "__main__":
                        main()
                """)
            
            # Créer le module matrix_operations
            os.makedirs(os.path.join(temp_dir, "matrix_operations"), exist_ok=True)
            with open(os.path.join(temp_dir, "matrix_operations", "__init__.py"), 'w') as f:
                f.write("")
            
            with open(os.path.join(temp_dir, "matrix_operations", "multiplication.py"), 'w') as f:
                f.write("""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def multiply_matrices_block(matrix_a_rows, matrix_b_cols, algorithm='standard', result_shape=None):

    logger.info(f"Multiplication de matrices avec l'algorithme {algorithm}")
    
    # Vérifier que les matrices sont compatibles pour la multiplication
    if matrix_a_rows.shape[1] != matrix_b_cols.shape[0]:
        raise ValueError(f"Les matrices ne sont pas compatibles pour la multiplication: "
                        f"A[{matrix_a_rows.shape}] x B[{matrix_b_cols.shape}]")
    
    if algorithm == 'strassen':
        # Implémenter l'algorithme de Strassen ici
        # Cette implémentation est simplifiée, en production, utiliser une bibliothèque optimisée
        logger.info("Utilisation de l'algorithme de Strassen (simplifié)")
        result = strassen_multiply(matrix_a_rows, matrix_b_cols)
    elif algorithm == 'winograd':
        # Implémenter l'algorithme de Winograd ici
        logger.info("Utilisation de l'algorithme de Winograd (fallback vers standard)")
        result = np.matmul(matrix_a_rows, matrix_b_cols)
    else:
        # Multiplication standard avec numpy
        logger.info("Utilisation de l'algorithme standard")
        result = np.matmul(matrix_a_rows, matrix_b_cols)
    
    # Si des dimensions spécifiques sont demandées, extraire le bloc correspondant
    if result_shape:
        rows, cols = result_shape
        if rows > 0 and cols > 0:
            # S'assurer que le résultat a au moins les dimensions demandées
            if result.shape[0] >= rows and result.shape[1] >= cols:
                result = result[:rows, :cols]
            else:
                raise ValueError(f"Le résultat calculé [{result.shape}] est plus petit "
                               f"que les dimensions demandées [{rows}, {cols}]")
    
    return result

def strassen_multiply(A, B):
    # Implémentation simplifiée de l'algorithme de Strassen pour la multiplication matricielle
    
    # Cette implémentation est un exemple et n'est pas optimisée pour la production.
    # En production, utilisez une bibliothèque comme numpy qui implémente déjà des 
    # algorithmes optimisés pour la multiplication matricielle.
    
    # Args:
    #     A (numpy.ndarray): Première matrice
    #     B (numpy.ndarray): Seconde matrice
        
    # Returns:
    #     numpy.ndarray: Résultat de la multiplication
    # Pour cet exemple, nous utilisons simplement numpy
    # Une véritable implémentation de Strassen serait plus complexe
    return np.matmul(A, B)
""")
            
            # Copier le module utils (réutiliser celui de l'addition)
            os.makedirs(os.path.join(temp_dir, "utils"), exist_ok=True)
            with open(os.path.join(temp_dir, "utils", "__init__.py"), 'w') as f:
                f.write("")
            
            with open(os.path.join(temp_dir, "utils", "messaging.py"), 'w') as f:
                f.write("""
import json
import logging
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

def publish_status(broker_host, broker_port, task_id, volunteer_id, progress):

    try:
        client = mqtt.Client(f"worker-{volunteer_id}-{task_id}")
        client.connect(broker_host, broker_port, 60)
        
        # Préparer le message
        message = {
            "task_id": task_id,
            "volunteer_id": volunteer_id,
            "progress": progress,
            "status": "running" if progress < 100 else "completed"
        }
        
        # Publier le message
        result = client.publish("tasks/status", json.dumps(message), qos=1)
        
        # Attendre que le message soit envoyé
        result.wait_for_publish()
        
        # Déconnecter le client
        client.disconnect()
        
        logger.info(f"Statut publié: {progress}% pour la tâche {task_id}")
        
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la publication du statut: {e}")
        return False

def publish_results(broker_host, broker_port, task_id, volunteer_id, results, error_details=None):
    # Publie les résultats d'une tâche sur le broker MQTT
    
    # Args:
    #     broker_host (str): Hôte du broker MQTT
    #     broker_port (int): Port du broker MQTT
    #     task_id (str): ID de la tâche
    #     volunteer_id (str): ID du volontaire
    #     results (dict): Résultats de la tâche
    #     error_details (dict, optional): Détails d'erreur en cas d'échec
        
    # Returns:
    #     bool: True si la publication a réussi, False sinon
    try:
        client = mqtt.Client(f"worker-{volunteer_id}-{task_id}")
        client.connect(broker_host, broker_port, 60)
        
        # Préparer le message
        message = {
            "task_id": task_id,
            "volunteer_id": volunteer_id,
            "status": "completed" if error_details is None else "failed",
            "results_location": {
                "protocol": "http",
                "host": "{{WORKER_IP}}",  # Sera remplacé par l'IP du volontaire
                "port": 8080,
                "basePath": "/results",
                "files": ["/multiplication_result.json"]
            } if results else None,
            "error_details": error_details
        }
        
        # Publier le message
        result = client.publish("tasks/results", json.dumps(message), qos=1)
        
        # Attendre que le message soit envoyé
        result.wait_for_publish()
        
        # Déconnecter le client
        client.disconnect()
        
        logger.info(f"Résultats publiés pour la tâche {task_id}")
        
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la publication des résultats: {e}")
        return False
""")
            
            # Construction de l'image
            try:
                subprocess.run(
                    ["docker", "build", "-t", image_full_name, "."],
                    cwd=temp_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                logger.info(f"Image {image_full_name} construite avec succès")
                
                # Pousser l'image vers le registre Docker si activé
                if self.push_enabled:
                    self._push_image(image_full_name)
                
                return image_full_name
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Erreur lors de la construction de l'image: {e}")
                logger.error(f"STDOUT: {e.stdout.decode('utf-8')}")
                logger.error(f"STDERR: {e.stderr.decode('utf-8')}")
                raise ValueError(f"Erreur lors de la construction de l'image: {e}")
    
    def _push_image(self, image_name):
        """
        Pousse une image vers le registre Docker
        
        Args:
            image_name (str): Nom complet de l'image Docker
            
        Returns:
            bool: True si l'image a été poussée avec succès, False sinon
        """
        logger.info(f"Poussée de l'image {image_name} vers le registre {self.registry}")
        
        try:
            # Vérifier si le registre est local ou distant
            if self.registry.startswith('localhost') or self.registry.startswith('127.0.0.1'):
                logger.info("Registre local détecté, pas besoin d'authentification")
            else:
                # Pour un registre distant, essayer de s'authentifier
                # En production, utiliser un secret manager pour stocker les credentials
                registry_user = os.environ.get('DOCKER_REGISTRY_USER', '')
                registry_password = os.environ.get('DOCKER_REGISTRY_PASSWORD', '')
                
                if registry_user and registry_password:
                    logger.info(f"Authentification au registre {self.registry}")
                    subprocess.run(
                        ["docker", "login", self.registry, "-u", registry_user, "-p", registry_password],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    logger.info("Authentification réussie")
                else:
                    logger.warning("Identifiants de registre non trouvés, tentative de poussée sans authentification")
            
            # Pousser l'image
            subprocess.run(
                ["docker", "push", image_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Image {image_name} poussée avec succès")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de la poussée de l'image: {e}")
            logger.error(f"STDOUT: {e.stdout.decode('utf-8')}")
            logger.error(f"STDERR: {e.stderr.decode('utf-8')}")
            return False