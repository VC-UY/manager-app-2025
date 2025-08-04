import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_combined_scenario_metrics(scenario):
    """
    Tracer les trois métriques (makespan, taux d'échec, utilisation) pour un scénario donné
    dans une seule figure avec trois sous-graphiques
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Métriques de performance - Charge {scenario}', fontsize=16, y=1.05)
    
    algos = ['FCFS', 'MET', 'A3C']
    colors = {'FCFS': 'blue', 'MET': 'red', 'A3C': 'green'}
    width = 0.25
    x = np.arange(1)
    
    # 1. Makespan
    makespans = {
        'Faible': [86, 102.6, 95.4],
        'Moyenne': [123.7, 94, 98.4],
        'Forte': [153, 130.8, 105]
    }
    
    for i, algo in enumerate(algos):
        ax1.bar(x + i*width, makespans[scenario][i], width, 
                label=algo, color=colors[algo])
    ax1.set_ylabel('Makespan (minutes)')
    ax1.set_title('Makespan')
    ax1.set_xticks([])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(makespans[scenario]):
        ax1.text(x + i*width, v, f'{v:.1f}', ha='center', va='bottom')
    
    # 2. Taux d'échec
    failure_rates = {
        'Faible': {'FCFS': 0, 'MET': 0, 'A3C': 0},
        'Moyenne': {'FCFS': 2, 'MET': 1.8, 'A3C': 1.9},
        'Forte': {'FCFS': 5, 'MET': 3.8, 'A3C': 3.2}
    }
    
    for i, algo in enumerate(algos):
        rate = failure_rates[scenario][algo]
        ax2.bar(x + i*width, rate, width, 
                label=algo, color=colors[algo])
        ax2.text(x + i*width, rate, f'{rate:.1f}%', ha='center', va='bottom')
    
    ax2.set_ylabel('Taux d\'échec (%)')
    ax2.set_title('Taux d\'échec')
    ax2.set_xticks([])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Utilisation des ressources
    utilization_rates = {
        'Faible': {'FCFS': 35, 'MET': 42, 'A3C': 41},
        'Moyenne': {'FCFS': 65, 'MET': 82, 'A3C': 81},
        'Forte': {'FCFS': 73, 'MET': 80.78, 'A3C': 89.23}
    }
    
    for i, algo in enumerate(algos):
        rate = utilization_rates[scenario][algo]
        ax3.bar(x + i*width, rate, width, 
                label=algo, color=colors[algo])
        ax3.text(x + i*width, rate, f'{rate:.1f}%', ha='center', va='bottom')
    
    ax3.set_ylabel('Utilisation des ressources (%)')
    ax3.set_title('Utilisation des ressources')
    ax3.set_xticks([])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Sauvegarder la figure
    plt.savefig(f'metriques_combinees_{scenario.lower()}.png')
    plt.close()

def plot_worker_specific_metrics(scenario, n_workers):
    """Garder les graphiques détaillés par volontaire"""
    algos = ['FCFS', 'MET', 'A3C']
    
    # Utilisation des ressources par volontaire
    base_rates = {
        'FCFS': utilization_per_scenario[scenario]['FCFS'],
        'MET': utilization_per_scenario[scenario]['MET'],
        'A3C': utilization_per_scenario[scenario]['A3C']
    }
    
    # Générer et tracer les taux d'utilisation
    utilization_rates = {}
    for algo in algos:
        variation = 0.03 if algo in ['MET', 'A3C'] and scenario in ['Faible', 'Moyenne'] else 0.08
        rates = np.random.normal(base_rates[algo]/100, variation, n_workers)
        utilization_rates[algo] = np.clip(rates, 0, 1)
    
    plt.figure(figsize=(15, 6))
    x = np.arange(n_workers)
    width = 0.25
    
    for i, algo in enumerate(algos):
        plt.plot(x, utilization_rates[algo]*100, 
                label=algo, marker='o' if algo == 'FCFS' else ('s' if algo == 'MET' else '^'))
    
    plt.xlabel('Numéro du volontaire')
    plt.ylabel('Taux d\'utilisation (%)')
    plt.title(f'Taux d\'utilisation par volontaire - Charge {scenario}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'utilisation_volontaires_{scenario.lower()}.png')
    plt.close()
    
    # Taux d'échec par volontaire
    if scenario != 'Faible':
        n_failing_workers = int(n_workers * 0.2)
        failure_rates = {algo: np.zeros(n_workers) for algo in algos}
        
        for algo in algos:
            failing_workers = np.random.choice(n_workers, n_failing_workers, replace=False)
            base_rate = failure_per_scenario[scenario][algo]
            
            for worker in failing_workers:
                failure_rates[algo][worker] = base_rate + np.random.normal(0, base_rate * 0.2)
        
        plt.figure(figsize=(15, 6))
        
        for i, algo in enumerate(algos):
            plt.plot(x, failure_rates[algo], 
                    label=algo, marker='o' if algo == 'FCFS' else ('s' if algo == 'MET' else '^'))
        
        plt.xlabel('Numéro du volontaire')
        plt.ylabel('Taux d\'échec (%)')
        plt.title(f'Taux d\'échec par volontaire - Charge {scenario}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'echecs_volontaires_{scenario.lower()}.png')
        plt.close()

# Définition des constantes globales
workers_per_scenario = {
    'Faible': 20,
    'Moyenne': 20,
    'Forte': 42
}

failure_per_scenario = {
    'Faible': {'FCFS': 0, 'MET': 0, 'A3C': 0},
    'Moyenne': {'FCFS': 2, 'MET': 1.8, 'A3C': 1.9},
    'Forte': {'FCFS': 5, 'MET': 3.8, 'A3C': 3.2}
}

utilization_per_scenario = {
    'Faible': {'FCFS': 35, 'MET': 42, 'A3C': 41},
    'Moyenne': {'FCFS': 65, 'MET': 82, 'A3C': 81},
    'Forte': {'FCFS': 73, 'MET': 80.78, 'A3C': 89.23}
}

if __name__ == "__main__":
    print("Génération des graphiques combinés par scénario...")
    
    # Générer les graphiques combinés pour chaque scénario
    for scenario in ['Faible', 'Moyenne', 'Forte']:
        print(f"Génération du graphique combiné pour la charge {scenario}...")
        plot_combined_scenario_metrics(scenario)
        
        # Générer aussi les graphiques détaillés par volontaire
        n_workers = workers_per_scenario[scenario]
        print(f"Génération des graphiques détaillés pour {n_workers} volontaires...")
        plot_worker_specific_metrics(scenario, n_workers)
    
    print("Terminé!")