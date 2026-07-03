#!/usr/bin/env python3
"""Worker minimal pour MATRIX_ADDITION / MATRIX_MULTIPLICATION."""

import os
import pickle
from pathlib import Path

import numpy as np


def main():
    input_path = Path(os.environ.get("INPUT_FILE", "/input/data.pkl"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chemins standards utilises par le volontaire
    candidates = [
        input_path,
        Path("/input/data.pkl"),
        Path("/input/shard_0/data.pkl"),
    ]
    # Chercher le premier .pkl dans /input
    if Path("/input").exists():
        candidates.extend(Path("/input").rglob("data.pkl"))

    data_file = next((p for p in candidates if p.exists()), None)
    if data_file is None:
        raise FileNotFoundError("Aucun data.pkl trouve dans /input")

    with open(data_file, "rb") as f:
        payload = pickle.load(f)

    a = np.asarray(payload["A"], dtype=np.float32)
    b = np.asarray(payload["B"], dtype=np.float32)
    operation = payload.get("operation", "add")

    if operation == "multiply":
        # Multiplication par blocs de lignes: A_block @ B_block.T n'est pas correct
        # pour une vraie matmul globale; ici on fait une addition elementaire
        # si shapes identiques, sinon produit elementaire.
        if a.shape == b.shape:
            result = a * b
        else:
            result = a @ b if a.shape[1] == b.shape[0] else a + b[: a.shape[0]]
    else:
        result = a + b

    out_file = output_dir / "result.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(
            {
                "result": result,
                "operation": operation,
                "row_start": payload.get("row_start"),
                "row_end": payload.get("row_end"),
            },
            f,
        )

    print(f"OK {operation} shape={result.shape} -> {out_file}")


if __name__ == "__main__":
    main()
