import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_prepare_data(algo_number, load_type, target_size):
    # Correspondance spéciale pour les noms de fichiers
    filename_map = {
        ('1', 'faible'): 'algo1_faible_charge.csv',
        ('2', 'faible'): 'algo2_faible_charge.csv',
        ('3', 'faible'): 'algo3_faible_charge.csv',
        ('1', 'moyenne'): 'algo1_charge_moyenne.csv',
        ('2', 'moyenne'): 'algo2_charge_moyenne.csv',
        ('3', 'moyenne'): 'algo3_charge_moyenne.csv',
        ('1', 'forte'): 'algo1_forte_charge.csv',
        ('2', 'forte'): 'algo2_forte_charge.csv',
        ('3', 'forte'): 'algo3_forte_charge.csv'
    }
    
    filename = filename_map.get((str(algo_number), load_type))
    try:
        df = pd.read_csv(filename)
        execution_times = df['execution_time'].values
        
        # Si on a plus de données que nécessaire, on prend les n premières
        if len(execution_times) > target_size:
            execution_times = execution_times[:target_size]
        # Si on a moins de données, on complète avec des zéros
        elif len(execution_times) < target_size:
            execution_times = np.pad(execution_times, (0, target_size - len(execution_times)))
            
        return execution_times
    except FileNotFoundError:
        print(f"Fichier {filename} non trouvé, utilisation de zéros")
        return np.zeros(target_size)

def plot_comparison(load_type, n_tasks):
    plt.figure(figsize=(12, 6))
    
    # Charger les données pour chaque algorithme
    algo1_times = load_and_prepare_data(1, load_type, n_tasks)
    algo2_times = load_and_prepare_data(2, load_type, n_tasks)
    algo3_times = load_and_prepare_data(3, load_type, n_tasks)
    
    # Créer l'axe des x
    x = np.arange(1, n_tasks + 1)
    
    # Tracer les courbes
    plt.plot(x, algo1_times, 'b-', label='Algorithme 1', marker='o')
    plt.plot(x, algo2_times, 'r-', label='Algorithme 2', marker='s')
    plt.plot(x, algo3_times, 'g-', label='Algorithme 3', marker='^')
    
    plt.xlabel('Numéro de tâche')
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.title(f'Comparaison des temps d\'exécution - Charge {load_type}')
    plt.legend()
    plt.grid(True)
    
    # Sauvegarder le graphique
    plt.savefig(f'comparaison_charge_{load_type}.png')
    print(f"Graphique sauvegardé : comparaison_charge_{load_type}.png")
    plt.close()

# Tracer les trois graphiques
print("Génération des graphiques...")
plot_comparison('faible', 7)
plot_comparison('moyenne', 25)
plot_comparison('forte', 60)
print("Terminé!")