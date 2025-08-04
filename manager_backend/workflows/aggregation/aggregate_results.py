import os
import pickle
import sys
import torch

def aggregate_results(results_dir, output_model_path):
    """
    Agrège les résultats des tâches pour produire un modèle final.
    """
    aggregated_model = None
    task_count = 0

    for file_name in os.listdir(results_dir):
        if file_name.endswith("_result.pkl"):
            file_path = os.path.join(results_dir, file_name)
            with open(file_path, "rb") as f:
                task_result = pickle.load(f)
                model_weights = task_result.get("model_weights")

                if aggregated_model is None:
                    aggregated_model = model_weights
                else:
                    for key in aggregated_model.keys():
                        aggregated_model[key] += model_weights[key]
                task_count += 1

    if aggregated_model is not None:
        # Moyenne des poids
        for key in aggregated_model.keys():
            aggregated_model[key] /= task_count

        # Sauvegarder le modèle agrégé
        torch.save(aggregated_model, output_model_path)
        print(f"[INFO] Modèle agrégé sauvegardé : {output_model_path}")
    else:
        print("[ERROR] Aucun résultat trouvé pour l'agrégation.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python aggregate_results.py <results_dir> <output_model_path>")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_model_path = sys.argv[2]
    aggregate_results(results_dir, output_model_path)