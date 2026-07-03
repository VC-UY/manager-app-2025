#!/usr/bin/env python3
"""
Simulateur malaria compatible avec les scenarios XML generes par VolunSys.
Modele de Ross-Macdonald (transmission vecteur-hote) — calcul scientifique reel.
"""

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_population(scenario_path: Path) -> int:
    tree = ET.parse(scenario_path)
    root = tree.getroot()
    # Gerer namespaces eventuels
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "demography":
            pop = elem.attrib.get("popSize")
            if pop:
                return max(100, int(float(pop)))
    return 1000


def find_scenario() -> Path:
    candidates = [
        Path("/input/scenario.xml"),
        Path("input/scenario.xml"),
        Path("scenario.xml"),
    ]
    if Path("/input").exists():
        candidates.extend(Path("/input").rglob("scenario.xml"))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Aucun scenario.xml trouve dans /input")


def ross_macdonald(population: int, days: int = 365):
    """
    Modele de Ross-Macdonald discret.
    Retourne incidence journaliere et prevalence finale.
    """
    # Parametres epidemiologiques typiques (zone endemique)
    a = 0.3          # taux de piqure par moustique et par jour
    b = 0.5          # proba transmission moustique -> humain
    c = 0.5          # proba transmission humain -> moustique
    mu = 0.1         # mortalite moustique / jour
    r = 0.05         # taux de guerison humain / jour
    m = 2.0          # densite moustiques par humain

    humans_infected = max(1, int(population * 0.05))
    mosquitoes_infected = max(1, int(population * m * 0.02))
    mosquitoes_total = int(population * m)

    daily_incidence = []
    for day in range(days):
        s_h = population - humans_infected
        force_h = a * b * (mosquitoes_infected / max(1, mosquitoes_total))
        new_infections = force_h * s_h
        recoveries = r * humans_infected
        humans_infected = max(0.0, humans_infected + new_infections - recoveries)
        humans_infected = min(float(population), humans_infected)

        s_m = mosquitoes_total - mosquitoes_infected
        force_m = a * c * (humans_infected / max(1, population))
        mosquitoes_infected = max(
            0.0,
            mosquitoes_infected
            + force_m * s_m
            - mu * mosquitoes_infected,
        )
        mosquitoes_infected = min(float(mosquitoes_total), mosquitoes_infected)

        daily_incidence.append(max(0.0, new_infections))

    prevalence = humans_infected / population
    total_cases = sum(daily_incidence)
    eir = a * m * (mosquitoes_infected / max(1, mosquitoes_total)) * 365
    return {
        "population": population,
        "days": days,
        "total_cases": total_cases,
        "prevalence": prevalence,
        "eir_annual": eir,
        "daily_incidence": daily_incidence,
    }


def main():
    scenario = find_scenario()
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    population = parse_population(scenario)
    days = int(os.environ.get("SIM_DAYS", "365"))
    result = ross_macdonald(population, days=days)

    out_file = output_dir / "output.txt"
    lines = [
        "VolunSys OpenMalaria-compatible simulation",
        f"scenario={scenario}",
        f"population={result['population']}",
        f"days={result['days']}",
        f"total_cases={result['total_cases']:.2f}",
        f"prevalence={result['prevalence']:.6f}",
        f"eir_annual={result['eir_annual']:.4f}",
        "day,incidence",
    ]
    for day, incidence in enumerate(result["daily_incidence"], start=1):
        lines.append(f"{day},{incidence:.6f}")

    out_file.write_text("\n".join(lines) + "\n")
    print(
        f"OK malaria pop={population} cases={result['total_cases']:.1f} "
        f"prev={result['prevalence']:.4f} -> {out_file}"
    )


if __name__ == "__main__":
    main()
