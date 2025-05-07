# backend/workflows/services.py
from utils.mongodb import get_collection
from bson.objectid import ObjectId
from datetime import datetime
import uuid
import logging
import os
import numpy as np
from django.conf import settings
from .models import WorkflowType, WorkflowStatus, TaskStatus

logger = logging.getLogger(__name__)

class WorkflowService:
    @staticmethod
    def create_workflow(data):
        """
        Crée un workflow matriciel dans la base de données
        """
        workflows, client = get_collection('workflows')
        
        # Obtenir l'utilisateur workflow_manager par défaut
        workflow_data = {
            'name': data.get('name'),
            'description': data.get('description', ''),
            'workflow_type': data.get('workflow_type'),
            'owner': 'workflow_manager',  # Utilisateur par défaut
            'status': 'CREATED',
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'priority': data.get('priority', 1),
            'volunteer_preferences': data.get('volunteer_preferences', []),
            'min_volunteers': data.get('min_volunteers', 1),
            'max_volunteers': data.get('max_volunteers', 100),
            'estimated_resources': data.get('estimated_resources', {}),
            'tags': data.get('tags', []),
            'metadata': data.get('metadata', {})
        }
        
        # Ajouter les données d'entrée spécifiques aux matrices si présentes
        input_data = data.get('input_data')
        if input_data:
            # Validation des matrices selon le type d'opération
            if workflow_data['workflow_type'] in [WorkflowType.MATRIX_ADDITION, WorkflowType.MATRIX_MULTIPLICATION]:
                WorkflowService._validate_matrix_input(workflow_data['workflow_type'], input_data)
            
            workflow_data['metadata']['input_data'] = input_data
        
        result = workflows.insert_one(workflow_data)
        workflow_data['_id'] = str(result.inserted_id)
        client.close()
        
        return workflow_data
    
    @staticmethod
    def _validate_matrix_input(workflow_type, input_data):
        """
        Valide les informations sur les matrices selon le type d'opération
        """
        matrix_a = input_data.get('matrix_a', {})
        matrix_b = input_data.get('matrix_b', {})
        
        if not matrix_a or not matrix_b:
            raise ValueError("Les informations sur les matrices A et B sont requises")
        
        # Vérifier les dimensions
        dim_a = matrix_a.get('dimensions', [])
        dim_b = matrix_b.get('dimensions', [])
        
        if len(dim_a) < 2 or len(dim_b) < 2:
            raise ValueError("Les dimensions des matrices doivent être spécifiées")
        
        # Pour l'addition, les dimensions doivent être identiques
        if workflow_type == WorkflowType.MATRIX_ADDITION:
            if dim_a != dim_b:
                raise ValueError("Pour l'addition, les dimensions des matrices doivent être identiques")
        
        # Pour la multiplication, le nombre de colonnes de A doit être égal au nombre de lignes de B
        elif workflow_type == WorkflowType.MATRIX_MULTIPLICATION:
            if dim_a[1] != dim_b[0]:
                raise ValueError("Pour la multiplication, le nombre de colonnes de A doit être égal au nombre de lignes de B")
    
    @staticmethod
    def get_workflow(workflow_id):
        """
        Récupère un workflow par son ID
        """
        workflows, client = get_collection('workflows')
        query = {'_id': ObjectId(workflow_id)}
        
        workflow = workflows.find_one(query)
        client.close()
        
        if workflow:
            workflow['_id'] = str(workflow['_id'])
        
        return workflow
    
    @staticmethod
    def get_workflows(status=None, workflow_type=None, tag=None):
        """
        Récupère la liste des workflows avec filtrage optionnel
        """
        workflows, client = get_collection('workflows')
        query = {}
        
        if status:
            query['status'] = status
        
        if workflow_type:
            query['workflow_type'] = workflow_type
        
        if tag:
            query['tags'] = {"$in": [tag]}
        
        result = list(workflows.find(query).sort("created_at", -1))
        client.close()
        
        # Conversion des ObjectId en chaînes pour sérialisation JSON
        for workflow in result:
            workflow['_id'] = str(workflow['_id'])
        
        return result
    
    @staticmethod
    def update_workflow(workflow_id, data):
        """
        Met à jour un workflow existant
        """
        workflows, client = get_collection('workflows')
        query = {'_id': ObjectId(workflow_id)}
        
        # Exclure les champs en lecture seule
        update_data = {k: v for k, v in data.items() if k not in ['_id', 'created_at']}
        update_data['updated_at'] = datetime.now()
        
        result = workflows.update_one(query, {'$set': update_data})
        client.close()
        
        return result.modified_count > 0
    
    @staticmethod
    def delete_workflow(workflow_id):
        """
        Supprime un workflow et ses fichiers associés
        """
        workflows, client = get_collection('workflows')
        query = {'_id': ObjectId(workflow_id)}
        
        # Récupérer d'abord le workflow pour nettoyer les fichiers associés
        workflow = workflows.find_one(query)
        
        if workflow:
            # Supprimer les fichiers de matrices associés
            try:
                matrix_dir = os.path.join(settings.MEDIA_ROOT, 'matrices', str(workflow_id))
                if os.path.exists(matrix_dir):
                    import shutil
                    shutil.rmtree(matrix_dir)
            except Exception as e:
                logger.error(f"Erreur lors de la suppression des fichiers du workflow {workflow_id}: {e}")
        
        # Supprimer le workflow
        result = workflows.delete_one(query)
        client.close()
        
        return result.deleted_count > 0
    
    @staticmethod
    def submit_workflow(workflow_id):
        """
        Soumet un workflow pour exécution et prépare les fichiers de matrices
        """
        workflows, client = get_collection('workflows')
        query = {'_id': ObjectId(workflow_id)}
        
        # Vérifier l'état actuel du workflow
        workflow = workflows.find_one(query)
        if not workflow or workflow.get('status') != WorkflowStatus.CREATED:
            client.close()
            return False, "Le workflow doit être à l'état CREATED pour être soumis"
        
        try:
            # Préparation des fichiers de matrices si nécessaire
            if workflow.get('workflow_type') in [WorkflowType.MATRIX_ADDITION, WorkflowType.MATRIX_MULTIPLICATION]:
                WorkflowService._prepare_matrix_files(workflow_id, workflow)
            
            # Mise à jour du statut
            update_data = {
                'status': WorkflowStatus.SUBMITTED,
                'updated_at': datetime.now(),
                'submitted_at': datetime.now()
            }
            
            result = workflows.update_one(query, {'$set': update_data})
            client.close()
            
            if result.modified_count > 0:
                return True, "Workflow soumis avec succès"
            else:
                return False, "Échec de la mise à jour du workflow"
                
        except Exception as e:
            # En cas d'erreur, marquer le workflow comme échoué
            update_data = {
                'status': WorkflowStatus.FAILED,
                'updated_at': datetime.now(),
                'error_details': str(e)
            }


            # publication dans le canal d'adord  pour le coordonateur
            
            workflows.update_one(query, {'$set': update_data})
            client.close()
            
            logger.error(f"Erreur lors de la soumission du workflow {workflow_id}: {e}")
            return False, f"Erreur lors de la soumission: {str(e)}"
    
    @staticmethod
    def _prepare_matrix_files(workflow_id, workflow):
        """
        Prépare les fichiers de matrices pour le traitement
        """
        if 'metadata' not in workflow or 'input_data' not in workflow['metadata']:
            raise ValueError("Les données d'entrée sont manquantes dans les métadonnées du workflow")
        
        input_data = workflow['metadata']['input_data']
        matrix_a_info = input_data.get('matrix_a', {})
        matrix_b_info = input_data.get('matrix_b', {})
        
        # Créer le répertoire pour stocker les matrices
        matrix_dir = os.path.join(settings.MEDIA_ROOT, 'matrices', str(workflow_id))
        os.makedirs(matrix_dir, exist_ok=True)
        
        # Traiter la matrice A
        if matrix_a_info.get('storage_type') == 'embedded' and 'data' in matrix_a_info:
            # La matrice est directement dans les métadonnées
            matrix_a_path = os.path.join(matrix_dir, 'matrix_a.npy')
            WorkflowService._save_matrix_to_file(matrix_a_info['data'], matrix_a_path)
            matrix_a_info['file_path'] = matrix_a_path
        
        # Traiter la matrice B
        if matrix_b_info.get('storage_type') == 'embedded' and 'data' in matrix_b_info:
            # La matrice est directement dans les métadonnées
            matrix_b_path = os.path.join(matrix_dir, 'matrix_b.npy')
            WorkflowService._save_matrix_to_file(matrix_b_info['data'], matrix_b_path)
            matrix_b_info['file_path'] = matrix_b_path
        
        # Mettre à jour les métadonnées du workflow
        workflows, client = get_collection('workflows')
        query = {'_id': ObjectId(workflow_id)}
        
        metadata = workflow['metadata']
        metadata['input_data']['matrix_a'] = matrix_a_info
        metadata['input_data']['matrix_b'] = matrix_b_info
        
        workflows.update_one(query, {'$set': {'metadata': metadata}})
        client.close()
    
    @staticmethod
    def _save_matrix_to_file(matrix_data, file_path):
        """
        Sauvegarde une matrice dans un fichier
        """
        # Convertir les données JSON en tableau numpy
        matrix = np.array(matrix_data)
        
        # Sauvegarder dans un fichier
        np.save(file_path, matrix)
        
        return file_path
    
    @staticmethod
    def update_workflow_status(workflow_id, status, details=None):
        """
        Met à jour le statut d'un workflow
        """
        workflows, client = get_collection('workflows')
        query = {'_id': ObjectId(workflow_id)}
        
        update_data = {
            'status': status,
            'updated_at': datetime.now()
        }
        
        # Ajout de détails supplémentaires si fournis
        if details:
            update_data.update(details)
        
        # Si le workflow est terminé, ajouter la date de fin
        if status == WorkflowStatus.COMPLETED:
            update_data['completed_at'] = datetime.now()
        
        result = workflows.update_one(query, {'$set': update_data})
        client.close()
        
        return result.modified_count > 0
    
    @staticmethod
    def get_workflow_tasks(workflow_id, status=None):
        """
        Récupère les tâches associées à un workflow avec filtrage optionnel
        """
        tasks, client = get_collection('tasks')
        query = {'workflow_id': str(workflow_id)}
        
        if status:
            query['status'] = status
        
        result = list(tasks.find(query))
        client.close()
        
        # Conversion des ObjectId en chaînes pour sérialisation JSON
        for task in result:
            task['_id'] = str(task['_id'])
        
        return result
    
    @staticmethod
    def aggregate_matrix_results(workflow_id):
        """
        Agrège les résultats des tâches matricielles
        """
        workflows, client = get_collection('workflows')
        query = {'_id': ObjectId(workflow_id)}
        
        workflow = workflows.find_one(query)
        if not workflow:
            client.close()
            raise ValueError(f"Workflow {workflow_id} introuvable")
        
        tasks, _ = get_collection('tasks')
        completed_tasks = list(tasks.find({
            'workflow_id': str(workflow_id),
            'status': TaskStatus.COMPLETED
        }))
        
        if not completed_tasks:
            client.close()
            raise ValueError("Aucune tâche complétée trouvée pour l'agrégation")
        
        # Créer le répertoire de résultats
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results', str(workflow_id))
        os.makedirs(result_dir, exist_ok=True)
        
        try:
            if workflow['workflow_type'] == WorkflowType.MATRIX_ADDITION:
                result_file = WorkflowService._aggregate_addition_results(workflow, completed_tasks, result_dir)
            elif workflow['workflow_type'] == WorkflowType.MATRIX_MULTIPLICATION:
                result_file = WorkflowService._aggregate_multiplication_results(workflow, completed_tasks, result_dir)
            else:
                client.close()
                raise ValueError(f"Type de workflow non supporté pour l'agrégation: {workflow['workflow_type']}")
            
            # Mettre à jour le workflow avec le résultat
            metadata = workflow.get('metadata', {})
            metadata['result_file'] = result_file
            
            workflows.update_one(query, {'$set': {
                'status': WorkflowStatus.COMPLETED,
                'updated_at': datetime.now(),
                'completed_at': datetime.now(),
                'metadata': metadata
            }})
            
            client.close()
            return result_file
            
        except Exception as e:
            logger.error(f"Erreur lors de l'agrégation des résultats: {e}")
            
            # Marquer le workflow comme échoué
            workflows.update_one(query, {'$set': {
                'status': WorkflowStatus.FAILED,
                'updated_at': datetime.now(),
                'error_details': str(e)
            }})
            
            client.close()
            raise
    
    @staticmethod
    def _aggregate_addition_results(workflow, completed_tasks, result_dir):
        """
        Agrège les résultats de l'addition matricielle
        """
        # Récupérer les dimensions finales depuis les métadonnées
        input_data = workflow.get('metadata', {}).get('input_data', {})
        matrix_a_info = input_data.get('matrix_a', {})
        dimensions = matrix_a_info.get('dimensions')
        
        if not dimensions or len(dimensions) < 2:
            raise ValueError("Dimensions de la matrice résultante non disponibles")
        
        # Créer une matrice résultante vide
        result_matrix = np.zeros(dimensions)
        
        # Remplir avec les résultats des tâches
        for task in completed_tasks:
            # Récupérer les coordonnées du bloc
            row_start = task.get('matrix_block_row_start')
            row_end = task.get('matrix_block_row_end')
            col_start = task.get('matrix_block_col_start')
            col_end = task.get('matrix_block_col_end')
            
            if None in [row_start, row_end, col_start, col_end]:
                continue
            
            # Charger le bloc résultat
            result_block_path = task.get('results', {}).get('block_file')
            if not result_block_path or not os.path.exists(result_block_path):
                continue
            
            try:
                block = np.load(result_block_path)
                result_matrix[row_start:row_end+1, col_start:col_end+1] = block
            except Exception as e:
                logger.error(f"Erreur lors du chargement du bloc {task['_id']}: {e}")
        
        # Sauvegarder la matrice résultante
        result_file = os.path.join(result_dir, 'final_matrix_addition.npy')
        np.save(result_file, result_matrix)
        
        return result_file
    
    @staticmethod
    def _aggregate_multiplication_results(workflow, completed_tasks, result_dir):
        """
        Agrège les résultats de la multiplication matricielle
        """
        # Récupérer les dimensions finales depuis les métadonnées
        input_data = workflow.get('metadata', {}).get('input_data', {})
        matrix_a_info = input_data.get('matrix_a', {})
        matrix_b_info = input_data.get('matrix_b', {})
        
        if not matrix_a_info.get('dimensions') or not matrix_b_info.get('dimensions'):
            raise ValueError("Dimensions des matrices d'entrée non disponibles")
        
        # Calculer les dimensions du résultat: (lignes de A) x (colonnes de B)
        dimensions = [matrix_a_info['dimensions'][0], matrix_b_info['dimensions'][1]]
        
        # Créer une matrice résultante vide
        result_matrix = np.zeros(dimensions)
        
        # Remplir avec les résultats des tâches
        for task in completed_tasks:
            # Récupérer les coordonnées du bloc
            row_start = task.get('matrix_block_row_start')
            row_end = task.get('matrix_block_row_end')
            col_start = task.get('matrix_block_col_start')
            col_end = task.get('matrix_block_col_end')
            
            if None in [row_start, row_end, col_start, col_end]:
                continue
            
            # Charger le bloc résultat
            result_block_path = task.get('results', {}).get('block_file')
            if not result_block_path or not os.path.exists(result_block_path):
                continue
            
            try:
                block = np.load(result_block_path)
                result_matrix[row_start:row_end+1, col_start:col_end+1] = block
            except Exception as e:
                logger.error(f"Erreur lors du chargement du bloc {task['_id']}: {e}")
        
        # Sauvegarder la matrice résultante
        result_file = os.path.join(result_dir, 'final_matrix_multiplication.npy')
        np.save(result_file, result_matrix)
        
        return result_file
    
    @staticmethod
    def get_matrix_statistics(workflow_id):
        """
        Récupère des statistiques sur les matrices d'un workflow
        """
        workflows, client = get_collection('workflows')
        query = {'_id': ObjectId(workflow_id)}
        
        workflow = workflows.find_one(query)
        client.close()
        
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} introuvable")
        
        stats = {
            "workflow_type": workflow.get('workflow_type'),
            "input_matrices": {},
            "result_matrix": None
        }
        
        # Statistiques sur les matrices d'entrée
        input_data = workflow.get('metadata', {}).get('input_data', {})
        
        for matrix_key in ['matrix_a', 'matrix_b']:
            matrix_info = input_data.get(matrix_key, {})
            file_path = matrix_info.get('file_path')
            
            if file_path and os.path.exists(file_path):
                try:
                    matrix = np.load(file_path)
                    stats["input_matrices"][matrix_key] = {
                        "dimensions": matrix.shape,
                        "data_type": str(matrix.dtype),
                        "file_size": os.path.getsize(file_path),
                        "min": float(np.min(matrix)),
                        "max": float(np.max(matrix)),
                        "mean": float(np.mean(matrix)),
                        "std": float(np.std(matrix))
                    }
                except Exception as e:
                    stats["input_matrices"][matrix_key] = {
                        "error": f"Erreur lors de l'analyse: {str(e)}"
                    }
        
        # Statistiques sur la matrice résultante si disponible
        result_file = workflow.get('metadata', {}).get('result_file')
        if result_file and os.path.exists(result_file):
            try:
                result_matrix = np.load(result_file)
                stats["result_matrix"] = {
                    "dimensions": result_matrix.shape,
                    "data_type": str(result_matrix.dtype),
                    "file_size": os.path.getsize(result_file),
                    "min": float(np.min(result_matrix)),
                    "max": float(np.max(result_matrix)),
                    "mean": float(np.mean(result_matrix)),
                    "std": float(np.std(result_matrix))
                }
            except Exception as e:
                stats["result_matrix"] = {
                    "error": f"Erreur lors de l'analyse: {str(e)}"
                }
        
        return stats