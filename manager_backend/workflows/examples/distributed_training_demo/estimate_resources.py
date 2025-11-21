# backend/workflows/utils/resource_estimator.py
import os
import pickle
import json
import math
import numpy as np

def estimate_flops_memory(data, model_params=None):
    """
    Estime les FLOPS et la mémoire nécessaires pour l'entraînement d'un modèle ML.
    
    Args:
        data: Données d'entraînement
        model_params: Paramètres du modèle (optionnel)
        
    Returns:
        tuple: (flops, memory_bytes)
    """
    if model_params is None:
        model_params = {
            'input_size': 3072,  # 32x32x3 pour CIFAR-10
            'hidden_layers': [512, 256, 128],
            'output_size': 10,
            'batch_size': 64,
            'epochs': 10
        }
    
    # Nombre d'échantillons
    n_samples = len(data)
    
    # Taille des données d'entrée
    input_size = model_params.get('input_size', 3072)
    
    # Tailles des couches cachées
    hidden_layers = model_params.get('hidden_layers', [128])
    
    # Taille de la sortie
    output_size = model_params.get('output_size', 10)
    
    # Nombre d'époques
    epochs = model_params.get('epochs', 10)
    
    # Taille de batch
    batch_size = model_params.get('batch_size', 64)
    
    # Calcul des FLOPS (opérations à virgule flottante)
    flops = 0
    prev_size = input_size
    
    # Pour chaque couche cachée
    for layer_size in hidden_layers:
        # Forward pass: multiplications matricielles + biais + activation
        # Chaque neurone: (prev_size multiplications + prev_size additions + 1 addition de biais + 1 activation)
        forward_ops = layer_size * (2 * prev_size + 2)
        
        # Backward pass: environ 2x le forward pass pour la rétropropagation
        backward_ops = 2 * forward_ops
        
        # Total pour cette couche
        layer_ops = forward_ops + backward_ops
        
        # Ajouter au total
        flops += layer_ops
        prev_size = layer_size
    
    # Couche de sortie
    output_ops = output_size * (2 * prev_size + 2) * 3  # forward + backward + softmax
    flops += output_ops
    
    # Multiplier par le nombre d'échantillons et d'époques
    total_flops = flops * n_samples * epochs
    
    # Estimation de la mémoire (en octets)
    # Données d'entrée
    input_memory = n_samples * input_size * 4  # 4 octets par float32
    
    # Paramètres du modèle
    model_params_memory = 0
    prev_size = input_size
    for layer_size in hidden_layers:
        model_params_memory += (prev_size * layer_size + layer_size) * 4  # Poids + biais
        prev_size = layer_size
    model_params_memory += (prev_size * output_size + output_size) * 4  # Couche de sortie
    
    # Mémoire pour les gradients (environ égale à la mémoire des paramètres)
    gradient_memory = model_params_memory
    
    # Mémoire pour les activations (dépend de la taille de batch)
    activation_memory = batch_size * (input_size + sum(hidden_layers) + output_size) * 4
    
    # Mémoire totale
    total_memory = input_memory + model_params_memory + gradient_memory + activation_memory
    
    # Ajouter une marge de sécurité de 20%
    total_memory = int(total_memory * 1.2)
    
    return total_flops, total_memory

def estimate_resources(inputs_dir):
    """
    Estime les ressources nécessaires pour un workflow d'entraînement ML.
    
    Args:
        inputs_dir: Répertoire contenant les données d'entrée
        
    Returns:
        dict: Ressources estimées
    """
    total_flops = 0
    total_memory = 0
    shard_count = 0
    data_size = 0
    
    # Paramètres du modèle (à adapter selon le type de modèle)
    model_params = {
        'input_size': 3072,  # 32x32x3 pour CIFAR-10
        'hidden_layers': [512, 256, 128],
        'output_size': 10,
        'batch_size': 64,
        'epochs': 10
    }
    
    try:
        # Parcourir les shards de données
        for name in os.listdir(inputs_dir):
            shard_path = os.path.join(inputs_dir, name)
            data_file = os.path.join(shard_path, "data.pkl")
            if os.path.isfile(data_file):
                with open(data_file, "rb") as f:
                    data, _ = pickle.load(f)
                
                # Estimer les FLOPS et la mémoire pour ce shard
                flops, memory = estimate_flops_memory(data, model_params)
                
                # Ajouter au total
                total_flops += flops
                total_memory += memory
                data_size += os.path.getsize(data_file)
                shard_count += 1
    except Exception as e:
        # En cas d'erreur, fournir une estimation par défaut
        print(f"Erreur lors de l'estimation des ressources: {e}")
        total_flops = 5e12  # 5 TFLOPS par défaut
        total_memory = 4 * 1024 * 1024 * 1024  # 4 Go par défaut
        shard_count = 1
        data_size = 100 * 1024 * 1024  # 100 Mo par défaut
    
    # Convertir les FLOPS en GFLOPS
    gflops = total_flops / 1e9
    
    # Estimer le temps d'exécution (en secondes)
    # Hypothèse: 10 GFLOPS/s pour un CPU standard, 100 GFLOPS/s pour un GPU standard
    cpu_time = gflops / 10
    gpu_time = gflops / 100
    
    # Estimer les besoins en CPU et GPU
    cpu_cores = max(1, min(16, math.ceil(gflops / 1e10)))  # Entre 1 et 16 cœurs
    gpu_needed = gflops > 1e12  # GPU nécessaire si plus de 1000 GFLOPS
    
    # Convertir la mémoire en Mo
    memory_mb = total_memory / (1024 * 1024)
    
    # Estimer l'espace disque nécessaire (données + modèle + checkpoints)
    disk_space = data_size * 3  # 3x la taille des données
    
    return {
        "estimated_flops": total_flops,
        "estimated_gflops": gflops,
        "estimated_memory_bytes": total_memory,
        "estimated_memory_mb": memory_mb,
        "estimated_cpu_time_seconds": cpu_time,
        "estimated_gpu_time_seconds": gpu_time if gpu_needed else None,
        "estimated_cpu_cores": cpu_cores,
        "gpu_required": gpu_needed,
        "estimated_disk_space_bytes": disk_space,
        "estimated_disk_space_mb": disk_space / (1024 * 1024),
        "shards": shard_count,
        "data_size_bytes": data_size,
        "data_size_mb": data_size / (1024 * 1024)
    }

