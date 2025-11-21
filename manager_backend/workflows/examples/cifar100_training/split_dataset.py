# workflows/examples/cifar100_training/split_dataset.py

import pickle, os
import numpy as np

def split_dataset(shards=4, path="./data/inputs", dataset_path="./data", logger=None):
    try:
        if logger: logger.warning(f"Chargement du dataset CIFAR-100 depuis {dataset_path}")

        pkl_path = os.path.join(dataset_path, "cifar-100-python", "train")
        with open(pkl_path, "rb") as f:
            raw_data = pickle.load(f, encoding="latin1")

        data = raw_data["data"]  # shape (50000, 3072)
        labels = raw_data["fine_labels"]

        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)

        total = len(data)
        size = total // shards

        if logger: logger.warning(f"Découpage en {shards} shards, taille de shard ≈ {size}")

        for i in range(shards):
            shard_dir = os.path.join(path, f"shard_{i}")
            os.makedirs(shard_dir, exist_ok=True)

            shard_data = data[i*size:(i+1)*size]
            shard_labels = labels[i*size:(i+1)*size]

            output_file = os.path.join(shard_dir, "data.pkl")
            with open(output_file, "wb") as f:
                pickle.dump((shard_data, shard_labels), f)

            if logger: logger.warning(f"✅ Shard {i+1}/{shards} sauvegardé dans {output_file}")

        if logger: logger.warning("✅ Découpage terminé avec succès")
    except Exception as e:
        if logger: logger.error(f"❌ Erreur durant le découpage : {e}")
        raise
