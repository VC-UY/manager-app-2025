# backend/workflows/views.py

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.utils import timezone
from .models import Workflow, WorkflowStatus, WorkflowType
from .serializers import WorkflowSerializer, WorkflowDetailSerializer
import logging
import numpy as np
import json
import os
from django.conf import settings

from tasks.models import Task, TaskStatus
from tasks.serializers import TaskSerializer
from tasks.services.task_splitter import TaskSplitter
from docker.services.DockerService import DockerService
from communication.mqtt_client import MQTTClient
from utils.matrix_utils import save_matrix_to_file, validate_matrix_dimensions

# Configuration du logger
logger = logging.getLogger(__name__)

class WorkflowViewSet(viewsets.ModelViewSet):
    """
    API endpoint pour gérer les workflows d'opérations matricielles
    """
    queryset = Workflow.objects.all().order_by('-created_at')
    permission_classes = [AllowAny]  # Autoriser l'accès sans authentification
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return WorkflowDetailSerializer
        return WorkflowSerializer
    
    def get_queryset(self):
        """Filtrer les workflows par type, status, ou tag"""
        queryset = Workflow.objects.all().order_by('-created_at')
        
        # Filtrage par type de workflow
        workflow_type = self.request.query_params.get('type')
        if workflow_type:
            queryset = queryset.filter(workflow_type=workflow_type)
        
        # Filtrage par statut
        status = self.request.query_params.get('status')
        if status:
            queryset = queryset.filter(status=status)
        
        # Filtrage par tag
        tag = self.request.query_params.get('tag')
        if tag:
            queryset = queryset.filter(tags__contains=[tag])
        
        return queryset
    
    def create(self, request, *args, **kwargs):
        """
        Création d'un workflow avec validation spécifique pour les opérations matricielles
        """
        data = request.data
        
        # Validation spécifique pour les matrices
        workflow_type = data.get('workflow_type')
        input_data = data.get('input_data', {})
        
        if workflow_type and input_data:
            matrix_a = input_data.get('matrix_a', {})
            matrix_b = input_data.get('matrix_b', {})
            
            # Vérification de la présence des matrices
            if not matrix_a or not matrix_b:
                return Response(
                    {"error": "Les informations sur les matrices A et B sont requises"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validation des dimensions selon le type d'opération
            try:
                error_msg = validate_matrix_dimensions(
                    workflow_type, 
                    matrix_a.get('dimensions', []), 
                    matrix_b.get('dimensions', [])
                )
                if error_msg:
                    return Response({"error": error_msg}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response(
                    {"error": f"Erreur lors de la validation des matrices: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # Création standard avec le sérialiseur
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        
        # Log de création
        logger.info(f"Workflow créé: {serializer.data['id']} - Type: {workflow_type}")
        
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
    
    @action(detail=True, methods=['post'])
    def submit(self, request, pk=None):
        """
        Soumission du workflow pour exécution
        """
        workflow = self.get_object()
        
        if workflow.status != WorkflowStatus.CREATED:
            return Response(
                {"error": "Le workflow doit être à l'état CREATED pour être soumis"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Log du type de workflow
        logger.info(f"Soumission du workflow {workflow.id} de type {workflow.workflow_type}")
        
        try:
            # Préparer les fichiers matrices si nécessaire
            if workflow.workflow_type in [WorkflowType.MATRIX_ADDITION, WorkflowType.MATRIX_MULTIPLICATION]:
                self._prepare_matrix_files(workflow)
            
            # Valider et mettre à jour le statut
            workflow.status = WorkflowStatus.SUBMITTED
            workflow.submitted_at = timezone.now()
            workflow.save()
            
            # Lancer le découpage des tâches selon le type de workflow
            if workflow.workflow_type == WorkflowType.MATRIX_ADDITION:
                tasks = TaskSplitter.split_addition_workflow(workflow)
            elif workflow.workflow_type == WorkflowType.MATRIX_MULTIPLICATION:
                tasks = TaskSplitter.split_multiplication_workflow(workflow)
            else:
                return Response(
                    {"error": f"Type de workflow non supporté: {workflow.workflow_type}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Mettre à jour le workflow
            workflow.status = WorkflowStatus.SPLITTING
            workflow.save()
            
            return Response({
                "status": "success", 
                "message": f"Workflow soumis avec succès et découpé en {len(tasks)} tâches",
                "task_count": len(tasks)
            })
        except Exception as e:
            logger.error(f"Erreur lors de la soumission du workflow: {str(e)}")
            workflow.status = WorkflowStatus.FAILED
            workflow.save()
            return Response(
                {"error": f"Erreur lors de la soumission du workflow: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _prepare_matrix_files(self, workflow):
        """
        Prépare les fichiers de matrices pour le traitement
        """
        if not workflow.metadata or 'input_data' not in workflow.metadata:
            raise ValueError("Les données d'entrée sont manquantes dans les métadonnées du workflow")
        
        input_data = workflow.metadata['input_data']
        matrix_a_info = input_data.get('matrix_a', {})
        matrix_b_info = input_data.get('matrix_b', {})
        
        # Créer le répertoire pour stocker les matrices
        matrix_dir = os.path.join(settings.MEDIA_ROOT, 'matrices', str(workflow.id))
        os.makedirs(matrix_dir, exist_ok=True)
        
        # Traiter la matrice A
        if matrix_a_info.get('storage_type') == 'embedded' and 'data' in matrix_a_info:
            # La matrice est directement dans les métadonnées
            matrix_a_path = os.path.join(matrix_dir, 'matrix_a.npy')
            save_matrix_to_file(matrix_a_info['data'], matrix_a_path)
            matrix_a_info['file_path'] = matrix_a_path
        
        # Traiter la matrice B
        if matrix_b_info.get('storage_type') == 'embedded' and 'data' in matrix_b_info:
            # La matrice est directement dans les métadonnées
            matrix_b_path = os.path.join(matrix_dir, 'matrix_b.npy')
            save_matrix_to_file(matrix_b_info['data'], matrix_b_path)
            matrix_b_info['file_path'] = matrix_b_path
        
        # Mettre à jour les métadonnées
        workflow.metadata['input_data']['matrix_a'] = matrix_a_info
        workflow.metadata['input_data']['matrix_b'] = matrix_b_info
        workflow.save(update_fields=['metadata'])
    
    @action(detail=True, methods=['post'])
    def containerize(self, request, pk=None):
        """
        Crée des conteneurs Docker pour les tâches du workflow
        """
        workflow = self.get_object()
        
        if workflow.status not in [WorkflowStatus.SPLITTING, WorkflowStatus.SUBMITTED]:
            return Response(
                {"error": f"Le workflow doit être à l'état SPLITTING ou SUBMITTED, pas {workflow.status}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Récupérer toutes les tâches du workflow
        tasks = Task.objects.filter(workflow=workflow)
        
        if not tasks.exists():
            return Response(
                {"error": "Le workflow n'a pas de tâches à conteneuriser"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Conteneuriser les tâches selon le type de workflow
            if workflow.workflow_type == WorkflowType.MATRIX_ADDITION:
                TaskSplitter.containerize_addition_tasks(list(tasks))
            elif workflow.workflow_type == WorkflowType.MATRIX_MULTIPLICATION:
                TaskSplitter.containerize_multiplication_tasks(list(tasks))
            else:
                return Response(
                    {"error": f"Type de workflow non supporté pour la conteneurisation: {workflow.workflow_type}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Mettre à jour le statut du workflow
            workflow.status = WorkflowStatus.ASSIGNING
            workflow.save()
            
            return Response({
                "status": "success", 
                "message": f"{tasks.count()} tâches conteneurisées avec succès",
                "tasks": TaskSerializer(tasks, many=True).data
            })
        except Exception as e:
            logger.error(f"Erreur lors de la conteneurisation du workflow {workflow.id}: {str(e)}")
            return Response(
                {"error": f"Erreur lors de la conteneurisation: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def docker_images(self, request, pk=None):
        """
        Récupère les images Docker associées à un workflow
        """
        workflow = self.get_object()
        
        tasks = Task.objects.filter(workflow=workflow)
        
        images = []
        for task in tasks:
            if task.docker_image:
                matrix_info = None
                if task.matrix_block_row_start is not None:
                    matrix_info = {
                        "block_position": [
                            task.matrix_block_row_start,
                            task.matrix_block_row_end,
                            task.matrix_block_col_start,
                            task.matrix_block_col_end
                        ],
                        "dimensions": [
                            task.matrix_block_row_end - task.matrix_block_row_start + 1,
                            task.matrix_block_col_end - task.matrix_block_col_start + 1
                        ]
                    }
                
                images.append({
                    "task_id": str(task.id),
                    "task_name": task.name,
                    "image": task.docker_image,
                    "status": task.status,
                    "matrix_info": matrix_info
                })
        
        return Response({
            "workflow_id": str(workflow.id),
            "workflow_name": workflow.name,
            "workflow_type": workflow.workflow_type,
            "image_count": len(images),
            "images": images
        })
    
    @action(detail=True, methods=['post'])
    def push_images(self, request, pk=None):
        """
        Force la poussée des images Docker sur le registre
        """
        workflow = self.get_object()
        docker_service = DockerService()
        
        tasks = Task.objects.filter(workflow=workflow)
        tasks_with_images = [task for task in tasks if task.docker_image]
        
        if not tasks_with_images:
            return Response(
                {"error": "Aucune tâche avec une image Docker trouvée"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        results = []
        for task in tasks_with_images:
            try:
                success = docker_service.push_image(task.docker_image)
                results.append({
                    "task_id": str(task.id),
                    "image": task.docker_image,
                    "success": success
                })
            except Exception as e:
                results.append({
                    "task_id": str(task.id),
                    "image": task.docker_image,
                    "success": False,
                    "error": str(e)
                })
        
        # Mettre à jour le workflow si toutes les images sont poussées
        if all(r.get("success", False) for r in results):
            # Passer à l'étape suivante uniquement si le workflow est en ASSIGNING
            if workflow.status == WorkflowStatus.ASSIGNING:
                workflow.status = WorkflowStatus.PENDING
                workflow.save()
                
                # Notifier le coordinateur que les tâches sont prêtes
                mqtt_client = MQTTClient()
                workflow_info = {
                    "workflow_id": str(workflow.id),
                    "task_count": len(tasks_with_images),
                    "matrix_type": workflow.workflow_type
                }
                mqtt_client.publish("workflows/tasks_ready", json.dumps(workflow_info))
        
        return Response({
            "workflow_id": str(workflow.id),
            "pushed_count": sum(1 for r in results if r.get("success", False)),
            "total_count": len(results),
            "results": results
        })
    
    @action(detail=True, methods=['get'])
    def tasks(self, request, pk=None):
        """
        Récupérer toutes les tâches d'un workflow avec filtrage optionnel
        """
        workflow = self.get_object()
        
        # Filtrer par statut si spécifié
        status_filter = request.query_params.get('status')
        tasks = Task.objects.filter(workflow=workflow)
        
        if status_filter:
            tasks = tasks.filter(status=status_filter)
        
        serializer = TaskSerializer(tasks, many=True)
        
        # Ajouter des statistiques
        stats = {
            "total": tasks.count(),
            "status_counts": {
                status_name: tasks.filter(status=status_value).count()
                for status_value, status_name in TaskStatus.choices
            }
        }
        
        return Response({
            "tasks": serializer.data,
            "stats": stats
        })
    
    @action(detail=True, methods=['post'])
    def aggregate_results(self, request, pk=None):
        """
        Déclenchement manuel de l'agrégation des résultats
        """
        workflow = self.get_object()
        
        # Vérifier que le workflow est dans un état permettant l'agrégation
        if workflow.status not in [WorkflowStatus.RUNNING, WorkflowStatus.PARTIAL_FAILURE]:
            return Response(
                {"error": "Le workflow doit être en cours d'exécution ou en échec partiel pour agréger les résultats"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Vérifier si toutes les tâches sont terminées
        tasks = Task.objects.filter(workflow=workflow)
        incomplete_tasks = tasks.exclude(status__in=[TaskStatus.COMPLETED, TaskStatus.FAILED])
        
        if incomplete_tasks.exists():
            return Response({
                "warning": f"Certaines tâches ne sont pas terminées ({incomplete_tasks.count()}). Voulez-vous agréger les résultats partiels?",
                "incomplete_tasks": TaskSerializer(incomplete_tasks, many=True).data
            }, status=status.HTTP_202_ACCEPTED)
        
        try:
            # Mise à jour du statut
            workflow.status = WorkflowStatus.AGGREGATING
            workflow.save()
            
            # Lancer l'agrégation selon le type de workflow
            if workflow.workflow_type == WorkflowType.MATRIX_ADDITION:
                result_file = TaskSplitter.aggregate_addition_results(workflow)
            elif workflow.workflow_type == WorkflowType.MATRIX_MULTIPLICATION:
                result_file = TaskSplitter.aggregate_multiplication_results(workflow)
            else:
                return Response(
                    {"error": f"Type de workflow non supporté pour l'agrégation: {workflow.workflow_type}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Mise à jour avec le résultat final
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = timezone.now()
            
            # Stockage du chemin du fichier résultat dans les métadonnées
            if not workflow.metadata:
                workflow.metadata = {}
            workflow.metadata['result_file'] = result_file
            workflow.save()
            
            return Response({
                "status": "success",
                "message": "Résultats agrégés avec succès",
                "result_file": result_file
            })
        except Exception as e:
            logger.error(f"Erreur lors de l'agrégation des résultats: {str(e)}")
            
            # En cas d'échec, mettre le workflow en état d'échec
            workflow.status = WorkflowStatus.FAILED
            workflow.save()
            
            return Response(
                {"error": f"Erreur lors de l'agrégation des résultats: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def result_info(self, request, pk=None):
        """
        Récupérer les informations sur le résultat du workflow
        """
        workflow = self.get_object()
        
        if workflow.status != WorkflowStatus.COMPLETED:
            return Response(
                {"error": "Le workflow n'est pas encore terminé"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Récupération des informations sur le résultat
        result_file = workflow.metadata.get('result_file') if workflow.metadata else None
        
        if not result_file or not os.path.exists(result_file):
            return Response(
                {"error": "Fichier de résultat introuvable"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Récupérer les dimensions et autres infos pertinentes
        result_info = {}
        
        try:
            # Charger la matrice résultante pour obtenir ses dimensions
            result_matrix = np.load(result_file)
            result_info['dimensions'] = result_matrix.shape
            result_info['data_type'] = str(result_matrix.dtype)
            result_info['file_size'] = os.path.getsize(result_file)
            
            # Statistiques basiques sur la matrice
            result_info['statistics'] = {
                'min': float(np.min(result_matrix)),
                'max': float(np.max(result_matrix)),
                'mean': float(np.mean(result_matrix)),
                'std': float(np.std(result_matrix))
            }
            
            # Échantillon de la matrice (premiers éléments)
            sample_size = min(5, result_matrix.shape[0], result_matrix.shape[1])
            result_info['sample'] = result_matrix[:sample_size, :sample_size].tolist()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du résultat: {str(e)}")
            result_info['error'] = str(e)
        
        return Response({
            "workflow_id": str(workflow.id),
            "workflow_type": workflow.workflow_type,
            "result_file": result_file,
            "execution_time": (workflow.completed_at - workflow.submitted_at).total_seconds() if workflow.completed_at and workflow.submitted_at else None,
            "matrix_info": result_info
        })
    
    @action(detail=True, methods=['post'])
    def pause(self, request, pk=None):
        """Mettre en pause l'exécution du workflow"""
        workflow = self.get_object()
        
        if workflow.status != WorkflowStatus.RUNNING:
            return Response(
                {"error": "Seul un workflow en cours d'exécution peut être mis en pause"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Mettre à jour le statut
            workflow.status = WorkflowStatus.PAUSED
            workflow.save()
            
            # Notifier les volontaires via MQTT
            mqtt_client = MQTTClient()
            mqtt_client.publish(f"workflows/{workflow.id}/pause", json.dumps({"workflow_id": str(workflow.id)}))
            
            return Response({"status": "success", "message": "Workflow mis en pause"})
        except Exception as e:
            logger.error(f"Erreur lors de la mise en pause du workflow: {str(e)}")
            return Response(
                {"error": f"Erreur lors de la mise en pause: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def resume(self, request, pk=None):
        """Reprendre l'exécution d'un workflow en pause"""
        workflow = self.get_object()
        
        if workflow.status != WorkflowStatus.PAUSED:
            return Response(
                {"error": "Seul un workflow en pause peut être repris"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Mettre à jour le statut
            workflow.status = WorkflowStatus.RUNNING
            workflow.save()
            
            # Notifier les volontaires via MQTT
            mqtt_client = MQTTClient()
            mqtt_client.publish(f"workflows/{workflow.id}/resume", json.dumps({"workflow_id": str(workflow.id)}))
            
            return Response({"status": "success", "message": "Exécution du workflow reprise"})
        except Exception as e:
            logger.error(f"Erreur lors de la reprise du workflow: {str(e)}")
            return Response(
                {"error": f"Erreur lors de la reprise: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Annuler un workflow en cours"""
        workflow = self.get_object()
        
        if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            return Response(
                {"error": "Impossible d'annuler un workflow déjà terminé ou échoué"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Mettre à jour le statut
            previous_status = workflow.status
            workflow.status = WorkflowStatus.FAILED
            workflow.save()
            
            # Notifier les volontaires via MQTT
            mqtt_client = MQTTClient()
            mqtt_client.publish(f"workflows/{workflow.id}/cancel", json.dumps({
                "workflow_id": str(workflow.id),
                "previous_status": previous_status
            }))
            
            # Mettre à jour les tâches en cours
            running_tasks = Task.objects.filter(workflow=workflow, status=TaskStatus.RUNNING)
            for task in running_tasks:
                task.status = TaskStatus.FAILED
                task.error_details = {"reason": "Workflow annulé par l'utilisateur"}
                task.save()
            
            return Response({
                "status": "success", 
                "message": "Workflow annulé",
                "cancelled_tasks": running_tasks.count()
            })
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation du workflow: {str(e)}")
            return Response(
                {"error": f"Erreur lors de l'annulation: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )