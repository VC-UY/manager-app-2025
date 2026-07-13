import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compression import compress_model, decompress_model

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class JointSQTests(unittest.TestCase):
    def test_jointsq_compression_decompression(self):
        model = SimpleModel()
        
        # Initialiser avec des poids déterministes / connus
        with torch.no_grad():
            model.fc1.weight.copy_(torch.randn_like(model.fc1.weight))
            model.fc2.weight.copy_(torch.randn_like(model.fc2.weight))

        # Obtenir les poids originaux
        orig_weights = {k: v.clone() for k, v in model.state_dict().items()}

        # Compresser le modèle avec JointSQ (ratio de budget 0.05)
        ratio = 0.05
        compressed_bytes, meta = compress_model(model, method="jointsq", ratio=ratio)

        self.assertEqual(meta["method"], "jointsq")
        self.assertEqual(meta["ratio"], ratio)
        self.assertGreater(len(compressed_bytes), 0)

        # Créer un nouveau modèle vierge
        new_model = SimpleModel()

        # Décompresser dans le nouveau modèle
        decompress_model(new_model, compressed_bytes, meta)

        # Vérifier la reconstruction des formes et types
        for name, tensor in new_model.state_dict().items():
            orig = orig_weights[name]
            self.assertEqual(tensor.shape, orig.shape)
            self.assertEqual(tensor.dtype, orig.dtype)
            
            # Vérifier que le modèle reconstruit est fini
            self.assertTrue(torch.isfinite(tensor).all())

            # Si c'est un tenseur de float, vérifier qu'il n'est pas tout à fait nul (c.-à-d. pas corrompu à 0)
            if tensor.is_floating_point() and tensor.numel() > 0:
                # S'assurer que certains poids sont différents de 0
                self.assertGreater(torch.abs(tensor).sum().item(), 0.0)

    def test_jointsq_zero_size_tensor(self):
        # Vérifier que les tenseurs vides ou bizarres ne font pas planter l'algorithme
        class EmptyTensorModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.zeros(0))
        
        model = EmptyTensorModel()
        compressed_bytes, meta = compress_model(model, method="jointsq", ratio=0.05)
        self.assertEqual(meta["method"], "jointsq")
        
        new_model = EmptyTensorModel()
        decompress_model(new_model, compressed_bytes, meta)
        self.assertEqual(new_model.param.numel(), 0)


if __name__ == "__main__":
    unittest.main()
