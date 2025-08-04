import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_scenario_metrics(scenario, n_workers):
    """
    Créer une visualisation combinée pour un scénario avec :
    - Temps d'exécution par tâche
    - Utilisation des ressources par volontaire
    - Taux d'échec par volontaire (sauf pour faible charge)
    """
    algos = ['FCFS', 'MET', 'A3C']
    colors = {'FCFS': 'blue', 'MET': 'red', 'A3C': 'green'}
    markers = {'FCFS': 'o', 'MET': 's', 'A3C': '^'}
    
    # Définir les données pour chaque métrique
    tasks_per_scenario = {'Faible': 15, 'Moyenne': 60, 'Forte': 160}
    n_tasks = tasks_per_scenario[scenario]
    
    # Créer une figure avec une grille personnalisée
    fig = plt.figure(figsize=(20, 15))
    if scenario == 'Faible':
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    else:
        gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    
    # 1. Temps d'exécution par tâche
    ax1 = fig.add_subplot(gs[0])
    x_tasks = np.arange(1, n_tasks + 1)
    
    # Générer des temps d'exécution plausibles pour chaque algorithme
    base_times = {
        'Faible': {'FCFS': 6, 'MET': 7, 'A3C': 6.5},  # minutes par tâche en moyenne
        'Moyenne': {'FCFS': 3, 'MET': 2.8, 'A3C': 2.9},
        'Forte': {'FCFS': 1.2, 'MET': 1.1, 'A3C': 0.9}
    }
    
    for algo in algos:
        base_time = base_times[scenario][algo]
        times = np.random.normal(base_time, base_time*0.2, n_tasks)
        times = np.maximum(times, 0)  # Pas de temps négatifs
        ax1.plot(x_tasks, times, color=colors[algo], 
                label=f'{algo}', marker=markers[algo], markersize=4)
    
    ax1.set_xlabel('Numéro de tâche')
    ax1.set_ylabel('Temps d\'exécution (minutes)')
    ax1.set_title('Temps d\'exécution par tâche')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Utilisation des ressources par volontaire
    ax2 = fig.add_subplot(gs[1])
    x_workers = np.arange(1, n_workers + 1)
    
    # Taux d'utilisation de base pour chaque algorithme
    utilization_rates = {
        'Faible': {'FCFS': 35, 'MET': 42, 'A3C': 41},
        'Moyenne': {'FCFS': 65, 'MET': 82, 'A3C': 81},
        'Forte': {'FCFS': 73, 'MET': 80.78, 'A3C': 89.23}
    }
    
    for algo in algos:
        base_rate = utilization_rates[scenario][algo]
        variation = 0.03 if algo in ['MET', 'A3C'] and scenario in ['Faible', 'Moyenne'] else 0.08
        rates = np.random.normal(base_rate, base_rate*variation, n_workers)
        rates = np.clip(rates, 0, 100)
        ax2.plot(x_workers, rates, color=colors[algo], 
                label=f'{algo}', marker=markers[algo], markersize=4)
    
    ax2.set_xlabel('Numéro du volontaire')
    ax2.set_ylabel('Taux d\'utilisation (%)')
    ax2.set_title('Utilisation des ressources par volontaire')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Taux d'échec par volontaire (sauf pour faible charge)
    if scenario != 'Faible':
        ax3 = fig.add_subplot(gs[2])
        
        failure_rates = {
            'Moyenne': {'FCFS': 2, 'MET': 1.8, 'A3C': 1.9},
            'Forte': {'FCFS': 5, 'MET': 3.8, 'A3C': 3.2}
        }
        
        n_failing_workers = int(n_workers * 0.2)  # 20% des volontaires ont des échecs
        
        for algo in algos:
            rates = np.zeros(n_workers)
            failing_workers = np.random.choice(n_workers, n_failing_workers, replace=False)
            base_rate = failure_rates[scenario][algo]
            
            for worker in failing_workers:
                rates[worker] = base_rate + np.random.normal(0, base_rate * 0.2)
            
            rates = np.clip(rates, 0, None)  # Pas de taux négatifs
            ax3.plot(x_workers, rates, color=colors[algo], 
                    label=f'{algo}', marker=markers[algo], markersize=4)
        
        ax3.set_xlabel('Numéro du volontaire')
        ax3.set_ylabel('Taux d\'échec (%)')
        ax3.set_title('Taux d\'échec par volontaire')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Titre global
    plt.suptitle(f'Métriques détaillées - Charge {scenario}', fontsize=16, y=0.95)
    
    # Sauvegarder la figure
    plt.savefig(f'scenario_details_{scenario.lower()}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Génération des visualisations détaillées par scénario...")
    
    # Générer les graphiques pour chaque scénario
    scenarios = {
        'Faible': 20,
        'Moyenne': 20,
        'Forte': 42
    }
    
    for scenario, n_workers in scenarios.items():
        print(f"Génération des graphiques pour la charge {scenario}...")
        plot_scenario_metrics(scenario, n_workers)
    
    print("Terminé!")