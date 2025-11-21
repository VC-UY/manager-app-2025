# split_dataset.py
import pickle, os, sys
from torchvision.datasets import CIFAR10
import logging

logger = logging.getLogger(__name__)



def split_dataset(shards=4, path:str="./data", dataset_path:str="./data", logger:logging.Logger = logger):
    try:
        # Essayer d'abord sans télécharger
        try:
            logger.warning(f"Tentative d'utilisation du dataset existant dans {dataset_path}")
            dataset = CIFAR10(dataset_path, train=True, download=False)
        except Exception as e:
            import traceback
            logger.warning(f"Dataset non trouvé, téléchargement en cours: {e}")
            logger.error(traceback.format_exc())
            # Si ça échoue, essayer avec téléchargement
            dataset = CIFAR10(dataset_path, train=True, download=True)
            logger.warning(f"Téléchargement du dataset terminé")
        
        size = len(dataset) // shards
        logger.warning(f"Debut du decouppage en {shards} shards, taille de chaque shard: {size} exemples")
        
        for i in range(shards):
            shard_dir = f"{path}/inputs/shard_{i}"
            os.makedirs(shard_dir, exist_ok=True)
            logger.warning(f"Création du répertoire {shard_dir}")
            
            data = dataset.data[i*size:(i+1)*size]
            labels = dataset.targets[i*size:(i+1)*size]
            
            output_file = f"{shard_dir}/data.pkl"
            with open(output_file, "wb") as f:
                pickle.dump((data, labels), f)
            logger.warning(f"Shard {i+1}/{shards} écrit dans {output_file}")
            
        logger.warning(f"Decouppage termine avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du split du dataset: {e}")
        raise




if __name__ == "__main__":
    split_dataset()
