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
    queryset = Workflow.objects.all()
    serializer_class = WorkflowSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Workflow.objects.filter(owner=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        import os
        from django.conf import settings

        workflow = serializer.save(owner=self.request.user)
        data_root = '/data' if os.path.isdir('/data') else str(getattr(settings, 'BASE_DIR', '/tmp'))
        base_dir = os.path.join(
            data_root,
            'workflow_data',
            str(self.request.user.id),
            str(workflow.id),
        )
        if not workflow.executable_path:
            workflow.executable_path = base_dir
        if not workflow.input_path:
            workflow.input_path = os.path.join(workflow.executable_path, 'inputs')
        if not workflow.output_path:
            workflow.output_path = os.path.join(workflow.executable_path, 'outputs')
        os.makedirs(workflow.executable_path, exist_ok=True)
        os.makedirs(workflow.input_path, exist_ok=True)
        os.makedirs(workflow.output_path, exist_ok=True)
        # Parametres de demo pour ML / OpenMalaria ; CUSTOM exige une vraie config
        metadata = workflow.metadata or {}
        if workflow.workflow_type == 'ML_TRAINING':
            # Partitionnement d'un jeu global (8–12 partitions typiques)
            metadata.setdefault('num_tasks', 8)
            metadata.setdefault('samples_per_shard', 6000)
            metadata.setdefault('epochs', 25)
            metadata.setdefault('paradigm', 'partition_train_aggregate')
        if workflow.workflow_type == 'OPEN_MALARIA':
            # Étude globale partitionnée (8–12 sous-populations)
            metadata.setdefault('num_tasks', 8)
            metadata.setdefault('total_population', 160000)
            metadata.setdefault('population_per_task', 20000)
            metadata.setdefault('simulation_days', 3650)
            metadata.setdefault('monte_carlo_runs', 12)
            metadata.setdefault('paradigm', 'partition_simulate_aggregate')
        if workflow.workflow_type in ('MATRIX_ADDITION', 'MATRIX_MULTIPLICATION'):
            metadata.setdefault('num_tasks', 8)
        if workflow.workflow_type == 'CUSTOM':
            metadata.setdefault('num_tasks', 8)
        if workflow.workflow_type == 'CUSTOM':
            from rest_framework.exceptions import ValidationError
            from workflows.custom_validation import validate_custom_metadata

            ok, err, metadata = validate_custom_metadata(metadata)
            if not ok:
                workflow.delete()
                raise ValidationError({'metadata': err})
        workflow.metadata = metadata
        workflow.save()


    def perform_update(self, serializer):
        serializer.save(owner=self.request.user)


def process_workflow_submission(workflow_id):
    """
    Logique de soumission d'un workflow (sans encapsulation DRF).
    Utilisable depuis le dispatcher et l'API view.
    """
    try:
        # Récupérer le workflow
        workflow = get_object_or_404(Workflow, id=workflow_id)
        import os
        from django.conf import settings
        if not workflow.executable_path:
            data_root = '/data' if os.path.isdir('/data') else str(getattr(settings, 'BASE_DIR', '/tmp'))
            workflow.executable_path = os.path.join(
                data_root,
                'workflow_data',
                str(workflow.owner_id or 'anon'),
                str(workflow.id),
            )
        if not workflow.output_path:
            workflow.output_path = os.path.join(workflow.executable_path, 'outputs')
        if not workflow.input_path:
            workflow.input_path = os.path.join(workflow.executable_path, 'inputs')
        os.makedirs(workflow.executable_path, exist_ok=True)
        os.makedirs(workflow.output_path, exist_ok=True)
        os.makedirs(workflow.input_path, exist_ok=True)
        workflow.save(update_fields=['executable_path', 'output_path', 'input_path', 'updated_at'])
        
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
                    ip_address = "127.0.0.1"
                    thread_logger.warning(
                        "IP locale indisponible, fallback vers %s pour le serveur de fichiers",
                        ip_address,
                    )
                thread_logger.info(f"Adresse IP du serveur: {ip_address}")
                # Construire l'URL du serveur de fichiers
                file_server_url = f"http://{ip_address}:{server_port}"
                
                # Notifier la fin du découpage
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': 'SPLIT_COMPLETED',
                    'message': f'Découpage terminé, {len(tasks) if tasks else 0} tâches créées'
                })
                
                # Assigner ou attendre un volontaire (jamais d'echec pour absence de volontaire)
                volunteers = response.get('volunteers') if isinstance(response, dict) else None
                from tasks.assignment import assign_and_publish

                assign_result = assign_and_publish(workflow, volunteers, server_port)
                thread_logger.info("Assignation: %s", assign_result)
                workflow.refresh_from_db()

                try:
                    from tasks.handlers import (
                        listen_for_task_accept,
                        listen_for_task_complete,
                        listen_for_task_status,
                        listen_task_progress,
                    )
                    listen_for_task_accept()
                    listen_for_task_complete()
                    listen_for_task_status()
                    listen_task_progress()
                except Exception as listen_err:
                    thread_logger.warning("Ecoute taches partielle: %s", listen_err)

                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': workflow.status,
                    'message': assign_result.get('message', 'Processus de soumission terminé'),
                })
                thread_logger.info(f"===== Fin du traitement asynchrone du workflow {workflow_id} =====")
                
            except Exception as e:
                thread_logger.error(f"ERREUR lors du traitement asynchrone du workflow: {e}")
                import traceback
                thread_logger.error(traceback.format_exc())

                try:
                    workflow.refresh_from_db()
                    if workflow.tasks.exists():
                        workflow.status = WorkflowStatus.PENDING
                        workflow.save(update_fields=['status', 'updated_at'])
                        notify_event('workflow_status_change', {
                            'workflow_id': str(workflow_id),
                            'status': 'PENDING',
                            'message': (
                                'Soumission OK. Taches creees, en attente de volontaires '
                                f'(detail: {e})'
                            ),
                        })
                    else:
                        workflow.status = WorkflowStatus.FAILED
                        workflow.save(update_fields=['status', 'updated_at'])
                        notify_event('workflow_status_change', {
                            'workflow_id': str(workflow_id),
                            'status': 'FAILED',
                            'message': f'Erreur lors du traitement: {str(e)}',
                        })
                except Exception as db_error:
                    thread_logger.error(f"Impossible de mettre a jour le statut: {db_error}")
                
                thread_logger.error(f"===== Fin du traitement asynchrone du workflow {workflow_id} =====")
        
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


@api_view(['POST'])
def submit_workflow_view(request, workflow_id):
    """View to submit a workflow for processing."""
    return process_workflow_submission(workflow_id)

    
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
                
                if user.check_password(password):
                    token, created = Token.objects.get_or_create(user=user)
                    print(f"[DEBUG] Connexion réussie pour: {user.email}, Token: {token.key}")

                    # Ne pas bloquer la reponse HTTP sur le coordinateur
                    import threading

                    def _sync_coordinator_login():
                        try:
                            from workflows.coordinator_sync import sync_manager_to_coordinator
                            from workflows.models import User as UserModel

                            ok, data = sync_manager_to_coordinator(
                                username=user.username,
                                email=user.email,
                                password=password,
                                first_name=user.first_name or '',
                                last_name=user.last_name or '',
                            )
                            if ok and data.get('manager_id'):
                                db_user = UserModel.objects.get(pk=user.pk)
                                db_user.remote_id = data['manager_id']
                                db_user.save(update_fields=['remote_id'])
                        except Exception as sync_err:
                            print(f"[WARN] Synchronisation coordinateur: {sync_err}")

                    threading.Thread(target=_sync_coordinator_login, daemon=True).start()

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


@api_view(['GET'])
def get_workflow_outputs(request, workflow_id):
    """
    Récupère la liste des fichiers de sortie d'un workflow.
    """
    import os

    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentification requise'}, status=401)

    try:
        workflow = get_object_or_404(Workflow, id=workflow_id, owner=request.user)

        if not workflow.output_path or not os.path.exists(workflow.output_path):
            return JsonResponse({'files': [], 'message': 'Aucun fichier de sortie'}, status=200)

        # Lister tous les fichiers dans le répertoire de sortie
        files = []
        for root, dirs, filenames in os.walk(workflow.output_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, workflow.output_path)
                files.append({
                    'name': filename,
                    'path': relative_path,
                    'size': os.path.getsize(file_path),
                    'modified': os.path.getmtime(file_path)
                })

        return JsonResponse({
            'files': files,
            'output_path': workflow.output_path,
            'workflow_id': str(workflow.id),
            'workflow_name': workflow.name,
            'status': workflow.status
        }, status=200)

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des fichiers de sortie: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
def download_workflow_output(request, workflow_id, file_path):
    """
    Télécharge un fichier de sortie d'un workflow.
    """
    import os
    from django.http import FileResponse, Http404

    try:
        workflow = get_object_or_404(Workflow, id=workflow_id, owner=request.user)

        if not workflow.output_path:
            raise Http404("Chemin de sortie non défini")

        # Construire le chemin complet du fichier
        full_path = os.path.join(workflow.output_path, file_path)

        # Vérifier que le fichier existe et est dans le répertoire de sortie (sécurité)
        if not os.path.exists(full_path):
            raise Http404("Fichier non trouvé")

        # Vérifier que le chemin est bien dans le répertoire de sortie (éviter path traversal)
        if not os.path.realpath(full_path).startswith(os.path.realpath(workflow.output_path)):
            raise Http404("Accès non autorisé")

        # Renvoyer le fichier
        response = FileResponse(open(full_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))
        return response

    except Http404:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du fichier: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
def download_workflow_outputs_zip(request, workflow_id):
    """
    Télécharge tous les fichiers de sortie d'un workflow dans une archive ZIP.
    """
    import os
    import zipfile
    import tempfile
    from django.http import FileResponse, Http404

    try:
        workflow = get_object_or_404(Workflow, id=workflow_id, owner=request.user)

        if not workflow.output_path or not os.path.exists(workflow.output_path):
            raise Http404("Aucun fichier de sortie")

        # Créer un fichier ZIP temporaire
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')

        with zipfile.ZipFile(temp_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(workflow.output_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, workflow.output_path)
                    zf.write(file_path, arcname)

        temp_file.close()

        # Renvoyer le fichier ZIP
        zip_filename = f"{workflow.name.replace(' ', '_')}_outputs.zip"
        response = FileResponse(
            open(temp_file.name, 'rb'),
            as_attachment=True,
            filename=zip_filename
        )

        # Nettoyer le fichier temporaire après l'envoi
        response.file_to_stream.close_callback = lambda: os.unlink(temp_file.name)

        return response

    except Http404:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'archive ZIP: {e}")
        return JsonResponse({'error': str(e)}, status=500)