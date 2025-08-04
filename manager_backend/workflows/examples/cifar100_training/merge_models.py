import torch
from glob import glob

def merge_models(input_path, output_path):
    """
    Merge multiple PyTorch models stored in different directories.

    Args:
        input_path (str): Path to the directory containing the models to merge.
        output_path (str): Path to the output file.
    """
    models = [torch.load(f) for f in glob(input_path + "/*/model.pt")]
    avg_model = models[0]
    for k in avg_model:
        for m in models[1:]:
            avg_model[k] += m[k]
        avg_model[k] /= len(models)

    torch.save(avg_model, output_path)
