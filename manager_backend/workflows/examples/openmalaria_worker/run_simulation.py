#!/usr/bin/env python3
"""
Simulation épidémiologique (partition d'une étude globale).

Lit partition.json + scenario.xml produits par le Manager :
- même étude globale (paramètres partagés)
- sous-population allouée à ce shard
- modèle individu-centré + réplicats Monte-Carlo (charge CPU réaliste)

Sortie: output.txt (métriques de partition) pour agrégation globale.
"""

from __future__ import annotations

import json
import math
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def find_input_file(name: str) -> Path | None:
    roots = [Path("/input"), Path("input"), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.exists():
            return direct
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def parse_population(scenario_path: Path) -> int:
    tree = ET.parse(scenario_path)
    root = tree.getroot()
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "demography":
            pop = elem.attrib.get("popSize")
            if pop:
                return max(100, int(float(pop)))
    return 1000


def load_partition() -> dict:
    path = find_input_file("partition.json")
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def individual_based_run(population: int, days: int, seed: int, params: dict) -> dict:
    """
    Modèle individu-centré (sous-échantillon représentatif de la sous-population).
    Complexité ~ O(agents * days) — charge significative pour des partitions réalistes.
    """
    rng = random.Random(seed)
    a = float(params.get("bite_rate", 0.3))
    b = float(params.get("transmission_mh", 0.5))
    r = float(params.get("recovery_rate", 0.05))
    m = float(params.get("mosquito_density", 2.0))

    # Cap mémoire/CPU: on simule jusqu'à max_agents agents représentatifs
    max_agents = int(params.get("max_agents", 15000))
    n = max(100, min(population, max_agents))
    scale = population / n

    infected = [rng.random() < 0.05 for _ in range(n)]
    daily_incidence = []
    total_new = 0.0

    for _day in range(days):
        prev_infected = sum(1 for x in infected if x)
        force = a * b * m * (prev_infected / n)
        new_today = 0
        for i in range(n):
            if infected[i]:
                if rng.random() < r:
                    infected[i] = False
            else:
                if rng.random() < force:
                    infected[i] = True
                    new_today += 1
        # Remonter à l'échelle de la sous-population réelle
        daily_incidence.append(new_today * scale)
        total_new += new_today * scale

    prevalence = sum(1 for x in infected if x) / n
    eir = a * m * prevalence * 365.0
    return {
        "population": population,
        "agents_simulated": n,
        "days": days,
        "total_cases": total_new,
        "prevalence": prevalence,
        "eir_annual": eir,
        "daily_incidence_sample": daily_incidence[:: max(1, days // 30)],
    }


def main():
    scenario = find_input_file("scenario.xml")
    if not scenario:
        raise FileNotFoundError("Aucun scenario.xml trouvé dans /input")

    partition = load_partition()
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    population = int(partition.get("population_size") or parse_population(scenario))
    days = int(
        os.environ.get("SIM_DAYS")
        or partition.get("simulation_days")
        or 3650
    )
    mc_runs = int(
        os.environ.get("MONTE_CARLO_RUNS")
        or partition.get("monte_carlo_runs")
        or 12
    )
    base_seed = int(partition.get("seed") or partition.get("partition_index") or 0)
    params = partition.get("epidemiology") or {}

    runs = []
    for run_id in range(mc_runs):
        runs.append(
            individual_based_run(
                population=population,
                days=days,
                seed=base_seed * 10_000 + run_id,
                params=params,
            )
        )

    mean_prev = sum(r["prevalence"] for r in runs) / len(runs)
    mean_cases = sum(r["total_cases"] for r in runs) / len(runs)
    mean_eir = sum(r["eir_annual"] for r in runs) / len(runs)
    # Intervalle de confiance empirique (approx.)
    prev_sorted = sorted(r["prevalence"] for r in runs)
    lo = prev_sorted[max(0, int(0.05 * len(prev_sorted)))]
    hi = prev_sorted[min(len(prev_sorted) - 1, int(0.95 * len(prev_sorted)))]

    result = {
        "study_id": partition.get("study_id"),
        "partition_index": partition.get("partition_index"),
        "population_offset": partition.get("population_offset"),
        "population": population,
        "days": days,
        "monte_carlo_runs": mc_runs,
        "total_cases": mean_cases,
        "prevalence": mean_prev,
        "prevalence_ci95_low": lo,
        "prevalence_ci95_high": hi,
        "eir_annual": mean_eir,
        "model": "individual_based_ross_macdonald",
    }

    out_file = output_dir / "output.txt"
    lines = [
        "VolunSys OpenMalaria partition simulation",
        f"study_id={result['study_id']}",
        f"partition_index={result['partition_index']}",
        f"population_offset={result['population_offset']}",
        f"scenario={scenario}",
        f"population={result['population']}",
        f"days={result['days']}",
        f"monte_carlo_runs={result['monte_carlo_runs']}",
        f"total_cases={result['total_cases']:.4f}",
        f"prevalence={result['prevalence']:.8f}",
        f"prevalence_ci95_low={result['prevalence_ci95_low']:.8f}",
        f"prevalence_ci95_high={result['prevalence_ci95_high']:.8f}",
        f"eir_annual={result['eir_annual']:.6f}",
        f"model={result['model']}",
    ]
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics_path = output_dir / "partition_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"OK partition={result['partition_index']} pop={population} "
        f"mc={mc_runs} days={days} prev={mean_prev:.4f} -> {out_file}"
    )


if __name__ == "__main__":
    main()
