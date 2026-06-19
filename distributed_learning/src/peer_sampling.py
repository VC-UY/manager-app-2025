"""
Service de Peer Sampling pour le Gossip Learning classique.
Sélectionne de manière aléatoire et dynamique les voisins à chaque round.
"""
import random
import logging
from typing import List, Dict

def get_peer_sample(my_mac: str, all_macs: List[str], k: int) -> List[str]:
    """
    Sélectionne k pairs de manière aléatoire parmi all_macs (en excluant my_mac).
    C'est la base du Peer Sampling dans le Gossip Learning classique.
    """
    candidates = [mac for mac in all_macs if mac != my_mac]
    if not candidates:
        logging.debug("[PeerSampling] Aucun candidat disponible pour l'échantillonnage")
        return []
    
    sample_size = min(k, len(candidates))
    sample = random.sample(candidates, sample_size)
    logging.debug(f"[PeerSampling] {my_mac} a échantillonné {len(sample)} pairs parmi {len(candidates)} candidats")
    return sample

def build_peer_sampling_table(my_mac: str, all_macs: List[str], k: int) -> Dict:
    """Retourne la table d'échantillonnage pour le débogage/stats."""
    sampled = get_peer_sample(my_mac, all_macs, k)
    return {
        "my_mac": my_mac,
        "k": k,
        "sampled_peers": sampled
    }
