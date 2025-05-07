# backend/results/result_aggregator.py
import logging
from utils.mongodb import get_collection
from bson.objectid import ObjectId
from datetime import datetime
import json
import numpy as np
import math

logger = logging.getLogger(__name__)

class ResultAggregator:
    """Service pour agréger les résultats des sous-tâches"""
    
    @staticmethod
    def check_task_completion(task_id):
        """Vérifie si toutes les sous-tâches d'une tâche sont terminées"""
        tasks, client = get_collection('tasks')
        
        # Récupérer la tâche parent
        parent_task = tasks.find_one({'_id': ObjectId(task_id)})
        if not parent_task:
            client.close()
            logger.error(f"Parent task {task_id} not found")
            return False
        
        # Récupérer toutes les sous-tâches
        subtasks = list(tasks.find({
            'parent_task_id': str(task_id),
            'is_subtask': True
        }))
        
        if not subtasks:
            client.close()
            logger.warning(f"No subtasks found for task {task_id}")
            return False
        
        # Vérifier si toutes les sous-tâches sont terminées
        total_subtasks = len(subtasks)
        completed_subtasks = sum(1 for s in subtasks if s.get('status') == 'COMPLETED')
        failed_subtasks = sum(1 for s in subtasks if s.get('status') == 'FAILED')
        
        logger.info(f"Task {task_id}: {completed_subtasks}/{total_subtasks} completed, {failed_subtasks}/{total_subtasks} failed")
        
        # Si toutes les sous-tâches sont terminées (avec succès ou échec)
        if completed_subtasks + failed_subtasks == total_subtasks:
            # Déterminer si la tâche a réussi ou échoué
            if failed_subtasks == total_subtasks:
                # Toutes les sous-tâches ont échoué
                tasks.update_one(
                    {'_id': ObjectId(task_id)},
                    {'$set': {
                        'status': 'FAILED',
                        'updated_at': datetime.now(),
                        'completed_at': datetime.now(),
                        'error_details': {
                            'message': 'All subtasks failed',
                            'failed_count': failed_subtasks
                        }
                    }}
                )
                client.close()
                
                try:
                    # Notifier le changement de statut
                    from communication.coordinator_client import CoordinatorClient
                    coordinator = CoordinatorClient()
                    coordinator.broadcast_task_update(
                        task_id=str(task_id),
                        workflow_id=parent_task['workflow_id'],
                        status='FAILED',
                        message='All subtasks failed'
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting task update: {e}")
                
                return False
            
            elif failed_subtasks > 0:
                # Certaines sous-tâches ont échoué, mais pas toutes
                tasks.update_one(
                    {'_id': ObjectId(task_id)},
                    {'$set': {
                        'status': 'PARTIAL_FAILURE',
                        'updated_at': datetime.now(),
                        'error_details': {
                            'message': 'Some subtasks failed',
                            'failed_count': failed_subtasks,
                            'completed_count': completed_subtasks
                        }
                    }}
                )
                
                try:
                    # Notifier le changement de statut
                    from communication.coordinator_client import CoordinatorClient
                    coordinator = CoordinatorClient()
                    coordinator.broadcast_task_update(
                        task_id=str(task_id),
                        workflow_id=parent_task['workflow_id'],
                        status='PARTIAL_FAILURE',
                        message=f'{failed_subtasks} subtasks failed, {completed_subtasks} completed'
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting task update: {e}")
                
                client.close()
                return True  # On continue avec l'agrégation malgré les échecs partiels
            
            else:
                # Toutes les sous-tâches ont réussi
                tasks.update_one(
                    {'_id': ObjectId(task_id)},
                    {'$set': {
                        'status': 'AGGREGATING',
                        'updated_at': datetime.now()
                    }}
                )
                
                try:
                    # Notifier le changement de statut
                    from communication.coordinator_client import CoordinatorClient
                    coordinator = CoordinatorClient()
                    coordinator.broadcast_task_update(
                        task_id=str(task_id),
                        workflow_id=parent_task['workflow_id'],
                        status='AGGREGATING',
                        message='Starting result aggregation'
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting task update: {e}")
                
                client.close()
                return True
        
        # Il reste des sous-tâches en cours
        client.close()
        return False
    
    @staticmethod
    def aggregate_data_processing_results(parent_task, subtasks):
        """Agrège les résultats de traitement de données"""
        # Trier les sous-tâches par plage de données
        completed_subtasks = [s for s in subtasks if s.get('status') == 'COMPLETED']
        completed_subtasks.sort(key=lambda s: s.get('metadata', {}).get('data_range', {}).get('start', 0))
        
        # Collecter et fusionner les résultats
        aggregated_data = []
        summary_stats = {
            'total_processed': 0,
            'min_value': float('inf'),
            'max_value': float('-inf'),
            'sum': 0,
            'count': 0
        }
        
        # Agréger les données et statistiques
        for subtask in completed_subtasks:
            result = subtask.get('results', {})
            data = result.get('data', [])
            stats = result.get('statistics', {})
            
            # Ajouter les données
            aggregated_data.extend(data)
            
            # Mettre à jour les statistiques
            summary_stats['total_processed'] += stats.get('processed_count', 0)
            summary_stats['min_value'] = min(summary_stats['min_value'], stats.get('min_value', float('inf')))
            summary_stats['max_value'] = max(summary_stats['max_value'], stats.get('max_value', float('-inf')))
            summary_stats['sum'] += stats.get('sum', 0)
            summary_stats['count'] += stats.get('count', 0)
        
        # Calculer la moyenne
        if summary_stats['count'] > 0:
            summary_stats['average'] = summary_stats['sum'] / summary_stats['count']
        else:
            summary_stats['average'] = 0
        
        # Si les valeurs min/max n'ont pas été mises à jour, les réinitialiser
        if summary_stats['min_value'] == float('inf'):
            summary_stats['min_value'] = 0
        if summary_stats['max_value'] == float('-inf'):
            summary_stats['max_value'] = 0
        
        return {
            'status': 'SUCCESS',
            'data': aggregated_data[:1000],  # Limiter la taille des données retournées
            'data_size': len(aggregated_data),
            'statistics': summary_stats,
            'processing_info': {
                'subtask_count': len(completed_subtasks),
                'total_subtasks': len(subtasks),
                'aggregation_time': datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def aggregate_scientific_computing_results(parent_task, subtasks):
        """Agrège les résultats de calcul scientifique"""
        # Trier les sous-tâches par plage d'itérations
        completed_subtasks = [s for s in subtasks if s.get('status') == 'COMPLETED']
        completed_subtasks.sort(key=lambda s: s.get('metadata', {}).get('iteration_range', {}).get('start', 0))
        
        # Collecter les résultats par itération
        all_results = {}
        convergence_data = []
        
        for subtask in completed_subtasks:
            result = subtask.get('results', {})
            iteration_results = result.get('iteration_results', {})
            subtask_convergence = result.get('convergence_data', [])
            
            # Fusionner les résultats d'itération
            for iter_key, iter_value in iteration_results.items():
                all_results[iter_key] = iter_value
            
            # Ajouter les données de convergence
            convergence_data.extend(subtask_convergence)
        
        # Trier les données de convergence par itération
        convergence_data.sort(key=lambda d: d.get('iteration', 0))
        
        # Déterminer si la convergence a été atteinte
        converged = any(d.get('converged', False) for d in convergence_data)
        final_error = convergence_data[-1].get('error') if convergence_data else None
        
        return {
            'status': 'SUCCESS',
            'iteration_results': all_results,
            'convergence_data': convergence_data,
            'final_result': {
                'converged': converged,
                'final_error': final_error,
                'last_iteration': len(convergence_data)
            },
            'computation_info': {
                'subtask_count': len(completed_subtasks),
                'total_subtasks': len(subtasks),
                'aggregation_time': datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def aggregate_rendering_results(parent_task, subtasks):
        """Agrège les résultats de rendu graphique"""
        # Déterminer la méthode de découpage utilisée
        first_subtask = subtasks[0] if subtasks else None
        if not first_subtask:
            return {'status': 'ERROR', 'message': 'No subtasks found'}
        
        split_method = first_subtask.get('metadata', {}).get('split_method')
        
        if split_method == 'frame_range':
            # Agrégation de rendu par frames
            completed_subtasks = [s for s in subtasks if s.get('status') == 'COMPLETED']
            completed_subtasks.sort(key=lambda s: s.get('metadata', {}).get('frame_range', {}).get('start', 0))
            
            # Collecter les chemins de fichiers générés
            frame_files = {}
            render_stats = {
                'total_render_time': 0,
                'average_frame_time': 0,
                'completed_frames': 0
            }
            
            for subtask in completed_subtasks:
                result = subtask.get('results', {})
                frame_range = subtask.get('metadata', {}).get('frame_range', {})
                start_frame = frame_range.get('start', 0)
                end_frame = frame_range.get('end', 0)
                
                # Ajouter les fichiers de frames
                for frame in range(start_frame, end_frame):
                    frame_key = f"frame_{frame:04d}"
                    if frame_key in result:
                        frame_files[frame_key] = result[frame_key]
                
                # Ajouter les statistiques de rendu
                render_stats['total_render_time'] += result.get('render_time', 0)
                render_stats['completed_frames'] += (end_frame - start_frame)
            
            # Calculer le temps moyen par frame
            if render_stats['completed_frames'] > 0:
                render_stats['average_frame_time'] = render_stats['total_render_time'] / render_stats['completed_frames']
            
            return {
                'status': 'SUCCESS',
                'frame_files': frame_files,
                'frame_count': len(frame_files),
                'render_statistics': render_stats,
                'rendering_info': {
                    'subtask_count': len(completed_subtasks),
                    'total_subtasks': len(subtasks),
                    'aggregation_time': datetime.now().isoformat()
                }
            }
            
        elif split_method == 'image_region':
            # Agrégation de rendu par régions d'image (tuiles)
            completed_subtasks = [s for s in subtasks if s.get('status') == 'COMPLETED']
            
            # Récupérer les informations sur l'image complète
            if not completed_subtasks:
                return {'status': 'ERROR', 'message': 'No completed subtasks'}
            
            resolution = completed_subtasks[0].get('metadata', {}).get('split_info', {}).get('resolution', {})
            width = resolution.get('width', 1920)
            height = resolution.get('height', 1080)
            
            # Créer un dictionnaire pour stocker les tuiles avec leurs coordonnées
            tiles = {}
            render_stats = {
                'total_render_time': 0,
                'tile_count': 0
            }
            
            for subtask in completed_subtasks:
                result = subtask.get('results', {})
                region = subtask.get('metadata', {}).get('region', {})
                
                # Ajouter la tuile avec ses coordonnées
                tile_data = result.get('tile_data')
                if tile_data:
                    tile_key = f"tile_{region.get('start_x')}_{region.get('start_y')}"
                    tiles[tile_key] = {
                        'data': tile_data,
                        'region': region
                    }
                
                # Ajouter les statistiques de rendu
                render_stats['total_render_time'] += result.get('render_time', 0)
                render_stats['tile_count'] += 1
            
            # Créer un identifiant pour le fichier d'image complet
            composite_image_id = f"render_{parent_task['_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            return {
                'status': 'SUCCESS',
                'composite_image_id': composite_image_id,
                'tile_data': tiles,
                'resolution': {'width': width, 'height': height},
                'render_statistics': render_stats,
                'rendering_info': {
                    'subtask_count': len(completed_subtasks),
                    'total_subtasks': len(subtasks),
                    'aggregation_time': datetime.now().isoformat()
                }
            }
            
        else:
            return {'status': 'ERROR', 'message': f'Unknown rendering split method: {split_method}'}
    
    @staticmethod
    def aggregate_machine_learning_results(parent_task, subtasks):
        """Agrège les résultats d'apprentissage automatique"""
        # Déterminer le mode (entraînement ou inférence)
        ml_mode = parent_task.get('metadata', {}).get('ml_mode', 'training')
        completed_subtasks = [s for s in subtasks if s.get('status') == 'COMPLETED']
        
        if ml_mode == 'training':
            # Agrégation des modèles partiels entraînés
            model_parts = []
            training_metrics = {
                'total_loss': 0,
                'total_accuracy': 0,
                'total_samples': 0,
                'training_time': 0
            }
            
            for subtask in completed_subtasks:
                result = subtask.get('results', {})
                
                # Ajouter les parties du modèle
                if 'model_part' in result:
                    model_parts.append(result['model_part'])
                
                # Collecter les métriques d'entraînement
                metrics = result.get('metrics', {})
                training_metrics['total_loss'] += metrics.get('loss', 0) * metrics.get('samples', 0)
                training_metrics['total_accuracy'] += metrics.get('accuracy', 0) * metrics.get('samples', 0)
                training_metrics['total_samples'] += metrics.get('samples', 0)
                training_metrics['training_time'] += result.get('training_time', 0)
            
            # Calculer les moyennes pondérées
            if training_metrics['total_samples'] > 0:
                training_metrics['avg_loss'] = training_metrics['total_loss'] / training_metrics['total_samples']
                training_metrics['avg_accuracy'] = training_metrics['total_accuracy'] / training_metrics['total_samples']
            
            return {
                'status': 'SUCCESS',
                'model_parts': model_parts,
                'model_parts_count': len(model_parts),
                'training_metrics': training_metrics,
                'ml_info': {
                    'mode': 'training',
                    'subtask_count': len(completed_subtasks),
                    'total_subtasks': len(subtasks),
                    'aggregation_time': datetime.now().isoformat()
                }
            }
            
        elif ml_mode == 'inference':
            # Agrégation des résultats d'inférence
            # Trier les sous-tâches par plage de données
            completed_subtasks.sort(key=lambda s: s.get('metadata', {}).get('inference_range', {}).get('start', 0))
            
            # Collecter les prédictions
            all_predictions = []
            inference_metrics = {
                'total_time': 0,
                'total_samples': 0
            }
            
            for subtask in completed_subtasks:
                result = subtask.get('results', {})
                
                # Ajouter les prédictions
                predictions = result.get('predictions', [])
                all_predictions.extend(predictions)
                
                # Collecter les métriques d'inférence
                inference_metrics['total_time'] += result.get('inference_time', 0)
                inference_metrics['total_samples'] += len(predictions)
            
            # Calculer le temps moyen par échantillon
            if inference_metrics['total_samples'] > 0:
                inference_metrics['avg_time_per_sample'] = inference_metrics['total_time'] / inference_metrics['total_samples']
            
            return {
                'status': 'SUCCESS',
                'predictions': all_predictions[:1000],  # Limiter pour éviter des réponses trop grandes
                'prediction_count': len(all_predictions),
                'inference_metrics': inference_metrics,
                'ml_info': {
                    'mode': 'inference',
                    'subtask_count': len(completed_subtasks),
                    'total_subtasks': len(subtasks),
                    'aggregation_time': datetime.now().isoformat()
                }
            }
            
        else:
            return {'status': 'ERROR', 'message': f'Unknown ML mode: {ml_mode}'}
    
    @staticmethod
    def aggregate_results(task_id):
        """Méthode principale pour agréger les résultats des sous-tâches"""
        # Vérifier si toutes les sous-tâches sont terminées
        ready = ResultAggregator.check_task_completion(task_id)
        if not ready:
            logger.warning(f"Task {task_id} not ready for aggregation")
            return False
        
        tasks, client = get_collection('tasks')
        
        # Récupérer la tâche parent
        parent_task = tasks.find_one({'_id': ObjectId(task_id)})
        if not parent_task:
            client.close()
            logger.error(f"Parent task {task_id} not found")
            return False
        
        # Récupérer les sous-tâches
        subtasks = list(tasks.find({
            'parent_task_id': str(task_id),
            'is_subtask': True
        }))
        
        if not subtasks:
            client.close()
            logger.warning(f"No subtasks found for task {task_id}")
            return False
        
        # Récupérer le type de workflow
        workflows, _ = get_collection('workflows')
        workflow = workflows.find_one({'_id': ObjectId(parent_task['workflow_id'])})
        
        if not workflow:
            client.close()
            logger.error(f"Workflow {parent_task['workflow_id']} not found")
            return False
        
        workflow_type = workflow.get('workflow_type')
        
        try:
            # Agréger les résultats selon le type de workflow
            if workflow_type == 'DATA_PROCESSING':
                result = ResultAggregator.aggregate_data_processing_results(parent_task, subtasks)
            elif workflow_type == 'SCIENTIFIC_COMPUTING':
                result = ResultAggregator.aggregate_scientific_computing_results(parent_task, subtasks)
            elif workflow_type == 'RENDERING':
                result = ResultAggregator.aggregate_rendering_results(parent_task, subtasks)
            elif workflow_type == 'MACHINE_LEARNING':
                result = ResultAggregator.aggregate_machine_learning_results(parent_task, subtasks)
            else:
                result = {'status': 'ERROR', 'message': f'Unknown workflow type: {workflow_type}'}
            
            # Mettre à jour la tâche parent avec les résultats agrégés
            tasks.update_one(
                {'_id': ObjectId(task_id)},
                {'$set': {
                    'status': 'COMPLETED' if result['status'] == 'SUCCESS' else 'FAILED',
                    'results': result,
                    'completed_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'progress': 100 if result['status'] == 'SUCCESS' else 0
                }}
            )
            
            # Mettre à jour le workflow si toutes les tâches sont terminées
            tasks_in_workflow = list(tasks.find({
                'workflow_id': parent_task['workflow_id'],
                'is_subtask': False,  # Seulement les tâches principales
                '_id': {'$ne': ObjectId(task_id)}  # Exclure la tâche courante
            }))
            
            all_completed = all(t.get('status') in ['COMPLETED', 'FAILED'] for t in tasks_in_workflow)
            if all_completed:
                # Déterminer le statut final du workflow
                failed_tasks = sum(1 for t in tasks_in_workflow if t.get('status') == 'FAILED')
                if failed_tasks > 0:
                    workflow_status = 'PARTIAL_FAILURE' if failed_tasks < len(tasks_in_workflow) else 'FAILED'
                else:
                    workflow_status = 'COMPLETED'
                
                workflows.update_one(
                    {'_id': ObjectId(parent_task['workflow_id'])},
                    {'$set': {
                        'status': workflow_status,
                        'completed_at': datetime.now(),
                        'updated_at': datetime.now()
                    }}
                )
                
                try:
                    # Notifier le changement de statut du workflow
                    from communication.coordinator_client import CoordinatorClient
                    coordinator = CoordinatorClient()
                    coordinator.broadcast_workflow_update(
                        workflow_id=parent_task['workflow_id'],
                        status=workflow_status,
                        message=f'Workflow {workflow_status.lower()}'
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting workflow update: {e}")
            
            try:
                # Notifier le changement de statut de la tâche
                from communication.coordinator_client import CoordinatorClient
                coordinator = CoordinatorClient()
                coordinator.broadcast_task_update(
                    task_id=str(task_id),
                    workflow_id=parent_task['workflow_id'],
                    status='COMPLETED' if result['status'] == 'SUCCESS' else 'FAILED',
                    progress=100 if result['status'] == 'SUCCESS' else 0,
                    message=f"Task {task_id} {result['status'].lower()}"
                )
            except Exception as e:
                logger.error(f"Error broadcasting task update: {e}")
            
            client.close()
            return True
            
        except Exception as e:
            logger.exception(f"Error aggregating results for task {task_id}: {e}")
            
            # Marquer la tâche comme échouée
            tasks.update_one(
                {'_id': ObjectId(task_id)},
                {'$set': {
                    'status': 'FAILED',
                    'updated_at': datetime.now(),
                    'error_details': {
                        'message': f'Result aggregation failed: {str(e)}',
                        'exception': str(e)
                    }
                }}
            )
            
            try:
                # Notifier l'échec
                from communication.coordinator_client import CoordinatorClient
                coordinator = CoordinatorClient()
                coordinator.broadcast_task_update(
                    task_id=str(task_id),
                    workflow_id=parent_task['workflow_id'],
                    status='FAILED',
                    message=f"Result aggregation failed: {str(e)}"
                )
            except Exception as e2:
                logger.error(f"Error broadcasting task update: {e2}")
            
            client.close()
            return False