import os
import numpy as np
import json
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

def validate_matrix_dimensions(workflow_type, matrix_a_dimensions, matrix_b_dimensions):
    """
    Valide les dimensions des matrices selon le type d'opération
    
    Args:
        workflow_type (str): Type d'opération ('MATRIX_ADDITION' ou 'MATRIX_MULTIPLICATION')
        matrix_a_dimensions (list): Dimensions de la matrice A [rows, cols]
        matrix_b_dimensions (list): Dimensions de la matrice B [rows, cols]
        
    Returns:
        str: Message d'erreur en cas d'incompatibilité, None si les dimensions sont valides
    """
    # Vérifier que les dimensions sont bien spécifiées
    if not matrix_a_dimensions or not matrix_b_dimensions:
        return "Les dimensions des matrices A et B doivent être spécifiées"
    
    # Vérifier que les dimensions sont des listes de 2 éléments
    if len(matrix_a_dimensions) != 2 or len(matrix_b_dimensions) != 2:
        return "Les dimensions des matrices doivent être [lignes, colonnes]"
    
    # Vérification selon le type d'opération
    if workflow_type in ['MATRIX_ADDITION', 'Addition de matrices']:
        # Pour l'addition, les dimensions doivent être identiques
        if matrix_a_dimensions != matrix_b_dimensions:
            return f"Pour l'addition matricielle, les matrices doivent avoir les mêmes dimensions. A: {matrix_a_dimensions}, B: {matrix_b_dimensions}"
    
    elif workflow_type in ['MATRIX_MULTIPLICATION', 'Multiplication de matrices']:
        # Pour la multiplication, le nombre de colonnes de A doit être égal au nombre de lignes de B
        if matrix_a_dimensions[1] != matrix_b_dimensions[0]:
            return f"Pour la multiplication matricielle, le nombre de colonnes de A ({matrix_a_dimensions[1]}) doit être égal au nombre de lignes de B ({matrix_b_dimensions[0]})"
    
    # Si tout est valide, retourner None (pas d'erreur)
    return None

def save_matrix_to_file(matrix_data, file_path):
    """
    Sauvegarde une matrice dans un fichier
    
    Args:
        matrix_data (list/ndarray): Données de la matrice
        file_path (str): Chemin du fichier de destination
        
    Returns:
        bool: True si la sauvegarde a réussi, False sinon
    """
    try:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convertir en ndarray si nécessaire
        if isinstance(matrix_data, list):
            matrix = np.array(matrix_data)
        else:
            matrix = matrix_data
        
        # Sauvegarder selon l'extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.npy':
            # Format binaire NumPy
            np.save(file_path, matrix)
        elif ext == '.txt' or ext == '.csv':
            # Format texte
            np.savetxt(file_path, matrix, delimiter=',')
        elif ext == '.json':
            # Format JSON
            with open(file_path, 'w') as f:
                json.dump(matrix.tolist(), f)
        else:
            # Par défaut, utiliser le format binaire NumPy
            np.save(file_path, matrix)
        
        logger.info(f"Matrice sauvegardée dans {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de la matrice: {e}")
        return False

def load_matrix_from_file(file_path):
    """
    Charge une matrice depuis un fichier
    
    Args:
        file_path (str): Chemin du fichier source
        
    Returns:
        ndarray: Matrice chargée
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
        
        # Charger selon l'extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.npy':
            # Format binaire NumPy
            matrix = np.load(file_path)
        elif ext == '.txt' or ext == '.csv':
            # Format texte
            matrix = np.loadtxt(file_path, delimiter=',')
        elif ext == '.json':
            # Format JSON
            with open(file_path, 'r') as f:
                matrix = np.array(json.load(f))
        else:
            # Par défaut, essayer le format binaire NumPy
            matrix = np.load(file_path)
        
        logger.info(f"Matrice chargée depuis {file_path}")
        return matrix
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la matrice: {e}")
        raise

def extract_matrix_block(matrix, row_start, row_end, col_start, col_end):
    """
    Extrait un bloc d'une matrice
    
    Args:
        matrix (ndarray): Matrice source
        row_start (int): Indice de début des lignes
        row_end (int): Indice de fin des lignes
        col_start (int): Indice de début des colonnes
        col_end (int): Indice de fin des colonnes
        
    Returns:
        ndarray: Bloc extrait
    """
    try:
        # Vérifier les indices
        if (row_start < 0 or row_end >= matrix.shape[0] or 
            col_start < 0 or col_end >= matrix.shape[1] or
            row_start > row_end or col_start > col_end):
            raise ValueError(f"Indices de bloc invalides: [{row_start}:{row_end}, {col_start}:{col_end}] pour une matrice de taille {matrix.shape}")
        
        # Extraire le bloc
        block = matrix[row_start:row_end+1, col_start:col_end+1]
        
        return block
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du bloc: {e}")
        raise

def get_matrix_info(matrix):
    """
    Récupère des informations sur une matrice
    
    Args:
        matrix (ndarray): Matrice à analyser
        
    Returns:
        dict: Informations sur la matrice
    """
    info = {
        'shape': matrix.shape,
        'dtype': str(matrix.dtype),
        'size_bytes': matrix.nbytes,
        'min': float(np.min(matrix)),
        'max': float(np.max(matrix)),
        'mean': float(np.mean(matrix)),
        'std': float(np.std(matrix))
    }
    
    return info

def get_matrix_memory_usage(rows, cols, dtype=np.float64):
    """
    Calcule l'utilisation mémoire d'une matrice
    
    Args:
        rows (int): Nombre de lignes
        cols (int): Nombre de colonnes
        dtype (np.dtype, optional): Type de données
        
    Returns:
        int: Taille en octets
    """
    # Créer une matrice vide pour obtenir la taille des éléments
    element_size = np.empty(1, dtype=dtype).itemsize
    
    # Calculer la taille totale
    total_size = rows * cols * element_size
    
    return total_size

def add_matrices(matrix_a, matrix_b):
    """
    Additionne deux matrices
    
    Args:
        matrix_a (ndarray): Première matrice
        matrix_b (ndarray): Seconde matrice
        
    Returns:
        ndarray: Résultat de l'addition
    """
    try:
        # Vérifier les dimensions
        if matrix_a.shape != matrix_b.shape:
            raise ValueError(f"Les matrices doivent avoir les mêmes dimensions. A: {matrix_a.shape}, B: {matrix_b.shape}")
        
        # Effectuer l'addition
        result = matrix_a + matrix_b
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de l'addition des matrices: {e}")
        raise

def multiply_matrices(matrix_a, matrix_b, algorithm='standard'):
    """
    Multiplie deux matrices
    
    Args:
        matrix_a (ndarray): Première matrice
        matrix_b (ndarray): Seconde matrice
        algorithm (str, optional): Algorithme à utiliser ('standard', 'strassen', 'winograd')
        
    Returns:
        ndarray: Résultat de la multiplication
    """
    try:
        # Vérifier les dimensions
        if matrix_a.shape[1] != matrix_b.shape[0]:
            raise ValueError(f"Dimensions incompatibles pour la multiplication. A: {matrix_a.shape}, B: {matrix_b.shape}")
        
        # Choisir l'algorithme
        if algorithm == 'strassen' and min(matrix_a.shape + matrix_b.shape) > 32:
            # Implémenter Strassen pour les grandes matrices
            # Note: Cette implémentation est un placeholder, l'algo complet serait plus complexe
            logger.info("Utilisation de l'algorithme de Strassen")
            result = np.matmul(matrix_a, matrix_b)
        elif algorithm == 'winograd':
            # Implémenter Winograd pour les applications spécifiques
            # Note: Ceci est un placeholder
            logger.info("Utilisation de l'algorithme de Winograd")
            result = np.matmul(matrix_a, matrix_b)
        else:
            # Multiplication standard
            logger.info("Utilisation de l'algorithme standard")
            result = np.matmul(matrix_a, matrix_b)
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de la multiplication des matrices: {e}")
        raise

def get_matrix_stats(matrix):
    """
    Calcule des statistiques sur une matrice
    
    Args:
        matrix (ndarray): Matrice à analyser
        
    Returns:
        dict: Statistiques calculées
    """
    stats = {
        'dimensions': matrix.shape,
        'elements': matrix.size,
        'memory_usage': matrix.nbytes,
        'min': float(np.min(matrix)),
        'max': float(np.max(matrix)),
        'mean': float(np.mean(matrix)),
        'median': float(np.median(matrix)),
        'std': float(np.std(matrix)),
        'sparsity': float(np.count_nonzero(matrix == 0) / matrix.size)
    }
    
    return stats