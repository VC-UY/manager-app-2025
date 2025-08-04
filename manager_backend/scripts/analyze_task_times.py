import json
import csv
import sys
from datetime import datetime
import argparse

def parse_datetime(dt_str):
    """Parse datetime string, handling NULL values"""
    if dt_str == "NULL":
        return None
    return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')

def calculate_execution_times(vt_file, t_file, output_file):
    # Charger les données
    with open(vt_file, 'r') as f:
        volunteer_tasks = json.load(f)
    
    with open(t_file, 'r') as f:
        tasks = json.load(f)
    
    # Créer un dictionnaire des dernières mises à jour des tâches
    task_updates = {task['id']: task['last_updated'] for task in tasks}
    
    # Préparer les résultats
    results = []
    
    # Analyser chaque tâche
    for vt in volunteer_tasks:
        task_id = vt['task_id']
        accepted_time = vt['accepted_at']
        
        if accepted_time != "NULL" and task_id in task_updates:
            last_update = task_updates[task_id]
            
            # Convertir les chaînes en objets datetime
            start_time = parse_datetime(accepted_time)
            end_time = parse_datetime(last_update)
            
            if start_time and end_time:
                # Calculer la durée en secondes
                duration = (end_time - start_time).total_seconds()
                
                results.append({
                    'task_id': task_id,
                    'execution_time': duration,
                    'start_time': accepted_time,
                    'end_time': last_update
                })
    
    # Écrire les résultats dans un fichier CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'execution_time', 'start_time', 'end_time'])
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description='Analyser les temps d\'exécution des tâches')
    parser.add_argument('vt_file', help='Fichier JSON des VolunteerTask')
    parser.add_argument('t_file', help='Fichier JSON des Task')
    parser.add_argument('-o', '--output', default='task_execution_times.csv', 
                        help='Fichier CSV de sortie (défaut: task_execution_times.csv)')
    
    args = parser.parse_args()
    
    calculate_execution_times(args.vt_file, args.t_file, args.output)
    print(f"Résultats écrits dans {args.output}")

if __name__ == "__main__":
    main()
