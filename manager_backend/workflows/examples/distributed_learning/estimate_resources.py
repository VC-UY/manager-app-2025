"""Estimation des ressources pour workflows DISTRIBUTED_LEARNING (PyTorch gossip)."""


def estimate_resources(metadata=None, input_path=None) -> dict:
    meta = metadata or {}
    n_vol = max(2, int(meta.get("n_volunteers") or 3))
    max_rounds = int(meta.get("max_rounds") or 10)
    model = str(meta.get("model") or "resnet18")

    ram_mb = 4096
    if model in ("resnet50", "resnet101", "resnet152", "vgg19"):
        ram_mb = 8192

    gossip_interval = int(meta.get("gossip_interval") or 30)
    estimated_seconds = max(600, max_rounds * (gossip_interval + 120))

    return {
        "cpu_cores": 2,
        "memory_mb": ram_mb,
        "disk_space_mb": 2048,
        "estimated_time_seconds": estimated_seconds,
        "n_volunteers": n_vol,
        "runtime": "vc-uyr",
        "notes": "Apprentissage distribué gossip AD-PSGD via runtime vc-uyr (sans Docker)",
    }
