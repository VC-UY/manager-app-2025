# backend/workflows/views.py
import json
from rest_framework import viewsets
from .models import Workflow, WorkflowStatus, User
from .serializers import WorkflowSerializer
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework import viewsets,  status
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from .serializers import WorkflowSerializer, RegisterSerializer
import traceback
from django.utils import timezone
import logging
from redis_communication.client import RedisClient


logger = logging.getLogger(__name__)

from .models import User


class WorkflowViewSet(viewsets.ModelViewSet):
    queryset = Workflow.objects.all().order_by('-created_at')
    serializer_class = WorkflowSerializer
    permission_classes = [AllowAny]


@api_view(['POST'])
def submit_workflow_view(request, workflow_id):
    """
    View to submit a workflow for processing.
    """
    try:
        # Récupérer le workflow
        workflow = get_object_or_404(Workflow, id=workflow_id)
        
        # Notifier le début du processus de soumission
        from websocket_service.client import notify_event
        notify_event('workflow_status_change', {
            'workflow_id': str(workflow_id),
            'status': 'SUBMITTING',
            'message': 'Soumission du workflow en cours...'
        })
        
        # Utiliser le gestionnaire de workflow pour soumettre le workflow
        from workflows.handlers import submit_workflow_handler
        success, response = submit_workflow_handler(str(workflow_id))
        logger.info(f"Submit workflow response: {response}")
        
        if not success:
            # Notifier l'échec de la soumission
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow_id),
                'status': 'SUBMISSION_FAILED',
                'message': f"Échec de la soumission: {response.get('message', 'Erreur inconnue')}"
            })
            return JsonResponse({'success': False, 'response': response}, status=400)
            
        # Soumission réussie, mettre à jour le statut et notifier
        workflow.status = WorkflowStatus.SPLITTING
        workflow.submitted_at = timezone.now()
        workflow.save()
        logger.info(f"Workflow {workflow_id} soumis avec succès")
        
        # Notifier la réussite de la soumission
        notify_event('workflow_status_change', {
            'workflow_id': str(workflow_id),
            'status': 'SPLITTING',
            'message': 'Soumission réussie, découpage en cours...'
        })
        
        # Réponse initiale au client HTTP
        response_data = {'success': True, 'message': 'Workflow soumis avec succès, traitement en cours'}
        
        # Lancer le découpage dans un thread séparé pour ne pas bloquer la réponse HTTP
        def process_workflow_async():
            thread_logger = logging.getLogger(f"workflow_thread_{workflow_id}")
            thread_logger.setLevel(logging.DEBUG)
            
            # Ajouter un handler pour afficher les logs dans la console
            if not thread_logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - [THREAD %(threadName)s] - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                thread_logger.addHandler(handler)
            
            thread_logger.info(f"===== Début du traitement asynchrone du workflow {workflow_id} =====")
            try:
                # Démarrer un serveur de fichiers pour ce workflow
                from tasks.file_server import start_file_server
                server_port = start_file_server(workflow)
                
                if not server_port:
                    thread_logger.error(f"Impossible de démarrer le serveur de fichiers pour le workflow {workflow_id}")
                    workflow.status = WorkflowStatus.FAILED
                    workflow.save()
                    notify_event('workflow_status_change', {
                        'workflow_id': str(workflow_id),
                        'status': 'FAILED',
                        'message': 'Impossible de démarrer le serveur de fichiers pour le workflow'
                    })
                    return
                
                thread_logger.info(f"Serveur de fichiers démarré sur le port {server_port}")
                
                # Découpage du workflow
                thread_logger.info(f"Lancement du découpage")
                from workflows.split_workflow import split_workflow
                tasks = split_workflow(str(workflow_id), workflow.workflow_type, thread_logger)
                thread_logger.info(f"Tasks: {len(tasks)} créées")
                
                # Préparer l'URL du serveur de fichiers
                from redis_communication.utils import get_local_ip
                ip_address = get_local_ip()
                if not ip_address:
                    thread_logger.error(f"Impossible de récupérer l'adresse IP locale du serveur")
                    raise Exception("Erreur lors de la récupération de l'adresse IP du serveur")
                thread_logger.info(f"Adresse IP du serveur: {ip_address}")
                # Construire l'URL du serveur de fichiers
                file_server_url = f"http://{ip_address}:{server_port}"
                
                # Notifier la fin du découpage
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': 'SPLIT_COMPLETED',
                    'message': f'Découpage terminé, {len(tasks) if tasks else 0} tâches créées'
                })
                
                # Mettre à jour le statut du workflow
                if response.get('volunteers'):
                    thread_logger.info(f"Volontaires disponibles: {len(response.get('volunteers'))}")
                    workflow.status = WorkflowStatus.ASSIGNING
                    workflow.save()
                    
                    # Notifier le début de l'assignation
                    thread_logger.info(f"Notification du début de l'assignation")
                    notify_event('workflow_status_change', {
                        'workflow_id': str(workflow_id),
                        'status': 'ASSIGNING',
                        'message': 'Attribution des tâches aux volontaires...'
                    })
                    
                    thread_logger.info(f"Lancement de l'assignation")
                    from tasks.scheduller import assign_workflow_to_volunteers
                    thread_logger.info(f"Volontaires disponibles: {response.get('volunteers')}")
                    
                    # Assigner les tâches
                    assignment_result = assign_workflow_to_volunteers(workflow, response.get('volunteers'))
                    thread_logger.info(f"Résultat de l'assignation: {assignment_result}")
                    
                    # Préparer les informations du serveur de fichiers pour chaque tâche
                    file_server_info = {
                        'base_url': file_server_url,
                        'workflow_id': str(workflow_id)
                    }
                    
                    # Préparer les données d'assignation pour les volontaires
                    assignments_by_volunteer = {}
                    
                    # Parcourir les assignations et les regrouper par volontaire
                    for task_id, volunteer_id in assignment_result.get('assignments', {}).items():
                        if volunteer_id not in assignments_by_volunteer:
                            assignments_by_volunteer[volunteer_id] = []
                        
                        # Récupérer la tâche
                        from tasks.models import Task
                        task = Task.objects.get(id=task_id)
                        
                        # Préparer les informations de fichiers d'entrée
                        input_files = []
                        if task.input_files:
                            for file_path in task.input_files:
                                input_files.append({
                                    'path': file_path,
                                    'size': 0  
                                })
                        
                        # Créer les données de la tâche pour ce volontaire
                        task_data = {
                            'task_id': str(task.id),
                            'name': task.name,
                            'parameters': task.parameters,
                            'input_data': {
                                'files': input_files,
                                'file_server': file_server_info
                            },
                            'estimated_execution_time': task.estimated_max_time,
                            'docker_information': task.docker_info or {}
                        }
                        
                        assignments_by_volunteer[volunteer_id].append(task_data)
                    
                    # Envoyer les assignations aux volontaires
                    redis_client = RedisClient.get_instance()
                    for volunteer_id, tasks_data in assignments_by_volunteer.items():
                        thread_logger.info(f"Envoi de {len(tasks_data)} tâches au volontaire {volunteer_id}")
                        
                        # Préparer le message d'assignation
                        assignment_message = {
                            'workflow_id': str(workflow_id),
                            'assignments': {
                                volunteer_id: tasks_data
                            }
                        }
                        
                        # Publier le message d'assignation
                        redis_client.publish('task/assignment', assignment_message)
                    
                    # Notifier la fin de l'assignation
                    notify_event('workflow_status_change', {
                        'workflow_id': str(workflow_id),
                        'status': 'ASSIGNED',
                        'message': 'Tâches attribuées avec succès'
                    })

                    # Démarrer le serveur de fichiers local pour servir les fichiers d'entrée
                    thread_logger.info(f"Démarrage du serveur de fichiers local")
                    from tasks.file_server import start_file_server
                    file_server_port = start_file_server(workflow)
                    thread_logger.info(f"Serveur de fichiers démarré sur le port {file_server_port}")
                    
                    # Préparer les informations de tâches complètes pour chaque volontaire
                    thread_logger.info(f"Préparation des informations de tâches pour chaque volontaire")
                    from tasks.models import Task
                    
                    # Récupérer l'adresse IP du serveur
                    from redis_communication.utils import get_local_ip
                    server_ip = get_local_ip()
                    if not server_ip:
                        thread_logger.error(f"Impossible de récupérer l'adresse IP locale du serveur")
                        raise Exception("Erreur lors de la récupération de l'adresse IP du serveur")
                    thread_logger.info(f"Adresse IP du serveur: {server_ip}")
                    
                    # Enrichir les informations d'assignation pour chaque volontaire
                    enriched_assignments = {}
                    for volunteer_id, task_list in assignment_result.items():
                        enriched_tasks = []
                        for task_info in task_list:
                            task_id = task_info['task_id']
                            task = Task.objects.get(id=task_id)
                            
                            # Ajouter les informations complètes de la tâche
                            enriched_task = {
                                'task_id': task_id,
                                'name': task.name,
                                'workflow_id': str(workflow.id),
                                'parameters': task.parameters,
                                'estimated_execution_time': task.estimated_max_time,
                                'input_data': {
                                    'files': task.input_files,
                                    'file_server': {
                                        'host': server_ip,
                                        'port': file_server_port,
                                        'base_url': f'http://{server_ip}:{file_server_port}'
                                    }
                                },
                                'input_data_size': task.input_size,
                                'docker_information': task.docker_info if hasattr(task, 'docker_info') else {}
                            }
                            enriched_tasks.append(enriched_task)
                        
                        enriched_assignments[volunteer_id] = enriched_tasks
                    
                    # Lancer l'ecoute des canaux task/accept et task/complete
                    thread_logger.info(f"Lancement de l'ecoute des canaux task/accept et task/complete")
                    from tasks.handlers import listen_for_task_accept, listen_for_task_complete, listen_for_task_status, listen_task_progress
                    import uuid
                    accept_success = listen_for_task_accept()
                    complete_success = listen_for_task_complete()
                    status_success = listen_for_task_status()
                    progress_success = listen_task_progress()
                    
                    if accept_success and complete_success and status_success and progress_success:
                        thread_logger.info(f"Souscription aux canaux task/accept et task/complete, task/status et task/progress réussie")
                    else:
                        thread_logger.warning(f"Problème lors de la souscription aux canaux: accept={accept_success}, complete={complete_success}, status={status_success}, progress={progress_success}")

                    # Publier sur le canal d'assignment
                    thread_logger.info(f"Publication sur le canal d'assignment")
                    client = RedisClient.get_instance()
                    from redis_communication.utils import get_manager_login_token
                    
                    client.publish('task/assignment',
                        {
                            'workflow_id': str(workflow_id),
                            'assignments': enriched_assignments,
                        },
                        str(uuid.uuid4()),
                        get_manager_login_token(),
                        'request'
                    )
                else:
                    thread_logger.info(f"Aucun volontaire reçu, lancement de l'écoute sur le canal d'assignment")
                    pubsub = RedisClient.get_instance()
                    pubsub.subscribe('workflow/assignment')
                    thread_logger.debug(f"Souscription au canal 'workflow/assignment' réussie")
                    
                    # Notifier l'attente de volontaires
                    thread_logger.info(f"Notification de l'attente de volontaires")
                    notify_event('workflow_status_change', {
                        'workflow_id': str(workflow_id),
                        'status': 'WAITING_VOLUNTEERS',
                        'message': 'En attente de volontaires disponibles...'
                    })
                    thread_logger.debug(f"Notification envoyée avec succès")
                
                # Lancer la fonction d'ecoute de la liste des volontaires chez le coordinateur
                thread_logger.info(f"Lancement de la fonction d'ecoute de la liste des volontaires chez le coordinateur")
                from workflows.handlers import listen_for_volunteers
                listen_for_volunteers(workflow_id)
                thread_logger.info(f"Fonction d'ecoute de la liste des volontaires chez le coordinateur lancée avec succès")
                
                # Notifier la fin du processus complet
                thread_logger.info(f"Notification de la fin du processus complet")
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': workflow.status,
                    'message': 'Processus de soumission terminé'
                })
                thread_logger.info(f"===== Fin du traitement asynchrone du workflow {workflow_id} =====")
                
            except Exception as e:
                thread_logger.error(f"ERREUR lors du traitement asynchrone du workflow: {e}")
                import traceback
                error_trace = traceback.format_exc()
                thread_logger.error(f"Stacktrace détaillé:\n{error_trace}")
                
                # Mettre à jour le statut du workflow dans la base de données
                try:
                    workflow.status = 'ERROR'
                    workflow.save()
                    thread_logger.info(f"Statut du workflow mis à jour à 'ERROR' dans la base de données")
                except Exception as db_error:
                    thread_logger.error(f"Impossible de mettre à jour le statut du workflow dans la base de données: {db_error}")
                
                # Notifier l'erreur
                thread_logger.info(f"Envoi de la notification d'erreur")
                try:
                    notify_event('workflow_status_change', {
                        'workflow_id': str(workflow_id),
                        'status': 'ERROR',
                        'message': f'Erreur lors du traitement: {str(e)}'
                    })
                    thread_logger.info(f"Notification d'erreur envoyée avec succès")
                except Exception as notify_error:
                    thread_logger.error(f"Impossible d'envoyer la notification d'erreur: {notify_error}")
                
                thread_logger.error(f"===== Fin du traitement asynchrone du workflow {workflow_id} avec ERREUR =====")
        
        # Démarrer le traitement asynchrone
        import threading
        thread_name = f"workflow-{workflow_id}-thread"
        thread = threading.Thread(target=process_workflow_async, name=thread_name)
        thread.daemon = True
        logger.info(f"Démarrage du thread '{thread_name}' pour le traitement asynchrone du workflow {workflow_id}")
        thread.start()
        logger.info(f"Thread '{thread_name}' démarré avec succès")
        
        # Retourner immédiatement une réponse au client HTTP
        return JsonResponse(response_data, status=200)
            
    except Workflow.DoesNotExist:
        logger.error(f"Workflow {workflow_id} non trouvé")
        return JsonResponse({'error': 'Workflow not found.'}, status=404)
    except Exception as e:
        import traceback
        logger.error(f"Erreur inattendue lors de la soumission du workflow {workflow_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Notifier l'erreur via WebSocket
        try:
            from websocket_service.client import notify_event
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow_id),
                'status': 'ERROR',
                'message': f'Erreur inattendue: {str(e)}'
            })
        except Exception:
            pass  # Ne pas échouer si la notification échoue
            
        return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)
    
class RegisterView(APIView):
    # TRÈS IMPORTANT: AllowAny est nécessaire pour permettre l'inscription!
    permission_classes = [AllowAny]
    authentication_classes = []  # Pas d'authentification nécessaire pour s'inscrire

    def post(self, request):

        try:
            # Si les données arrivent en tant que chaîne JSON, les parser
            if isinstance(request.data, str):
                data = json.loads(request.data)
            else:
                data = request.data
            
            serializer = RegisterSerializer(data=data)
            
            if serializer.is_valid():
                print("[DEBUG] Données d'inscription valides")
                
                # Création de l'utilisateur
                try:
                    user = serializer.save()
                    print(f"[DEBUG] Utilisateur créé avec succès: {user.email}")
                    
                    # Création du token
                    token, created = Token.objects.get_or_create(user=user)
                    
                    # Construction de la réponse
                    response_data = {
                        "user": {
                            "id": str(user.id),
                            "username": user.username,
                            "first_name": user.first_name,
                            "last_name": user.last_name,
                            "email": user.email
                        },
                        "token": token.key
                    }
                    
                    print(f"[DEBUG] Réponse d'inscription réussie: {response_data}")
                    return Response(response_data, status=status.HTTP_201_CREATED)
                except Exception as e:
                    print(f"[ERROR] Exception lors de la création de l'utilisateur: {str(e)}")
                    print(traceback.format_exc())
                    return Response({"error": f"Erreur lors de la création de l'utilisateur: {str(e)}"}, 
                                   status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                print(f"[DEBUG] Erreurs de validation: {serializer.errors}")
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            print(f"[ERROR] Exception non gérée dans RegisterView: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": f"Une erreur inattendue s'est produite: {str(e)}"}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LoginView(APIView):
    # TRÈS IMPORTANT: AllowAny est nécessaire pour permettre la connexion!
    permission_classes = [AllowAny]
    authentication_classes = []  # Pas d'authentification nécessaire pour se connecter

    def post(self, request):
        try:
            # Log des données pour le débogage (sans exposer le mot de passe)
            request_data = request.data.copy() if hasattr(request.data, 'copy') else dict(request.data)
            if 'password' in request_data:
                request_data['password'] = '********'
            
            print(f"[DEBUG] Données reçues pour la connexion: {request_data}")
            
            email = request.data.get('email')
            password = request.data.get('password')
            
            if not email or not password:
                return Response({
                    'error': 'Veuillez fournir un email et un mot de passe'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Récupérer l'utilisateur par email
            try:
                user = User.objects.get(email=email)
                print(f"[DEBUG] Utilisateur trouvé: {user.email}")
                
                if user.password == password:
                    # Connexion réussie
                    token, created = Token.objects.get_or_create(user=user)
                    print(f"[DEBUG] Connexion réussie pour: {user.email}, Token: {token.key}")
                    
                    return Response({
                        'token': token.key,
                        'user': {
                            'id': str(user.id),
                            'email': user.email,
                            'username': user.username,
                            'first_name': user.first_name,
                            'last_name': user.last_name
                        }
                    }, status=status.HTTP_200_OK)
                else:
                    # Mot de passe incorrect
                    print(f"[DEBUG] Mot de passe incorrect pour: {user.email}")
                    return Response({
                        'error': 'Identifiants incorrects'
                    }, status=status.HTTP_401_UNAUTHORIZED)
            except User.DoesNotExist:
                # Utilisateur non trouvé
                print(f"[DEBUG] Utilisateur non trouvé pour l'email: {email}")
                return Response({
                    'error': 'Identifiants incorrects'
                }, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            print(f"[ERROR] Exception non gérée dans LoginView: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": f"Une erreur inattendue s'est produite: {str(e)}"}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]  # Seuls les utilisateurs authentifiés peuvent se déconnecter

    def post(self, request):
        try:
            # Si utilisation de tokens, supprimer le token
            if request.auth and hasattr(request.auth, 'delete'):
                request.auth.delete()
                print(f"[DEBUG] Token supprimé pour l'utilisateur: {request.user.email}")
            
            return Response({"success": "Déconnexion réussie"}, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"[ERROR] Erreur lors de la déconnexion: {str(e)}")
            return Response({"error": "Erreur lors de la déconnexion"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)