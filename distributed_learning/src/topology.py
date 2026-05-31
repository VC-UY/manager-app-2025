"""
Topologie XOR Kademlia-style :
  1. Chaque IP est hachée avec SHA-256 → entier 64 bits.
  2. La distance entre deux nœuds = XOR de leurs hachés.
  3. Les k voisins les plus proches sont ceux à la plus petite distance XOR.
"""
import hashlib
import struct
import logging
from typing import List, Dict


def hash_ip(ip: str) -> int:
    """Hache une adresse IP en entier 64 bits non signé via SHA-256."""
    digest = hashlib.sha256(ip.encode("utf-8")).digest()
    return struct.unpack(">Q", digest[:8])[0]


def xor_distance(ip1: str, ip2: str) -> int:
    """Distance XOR entre deux adresses IP (basée sur leurs hachés)."""
    return hash_ip(ip1) ^ hash_ip(ip2)


def get_k_nearest_neighbors(my_ip: str,
                             all_ips: List[str],
                             k: int) -> List[str]:
    """
    Retourne les k voisins les plus proches de my_ip parmi all_ips,
    triés par distance XOR croissante.
    """
    candidates = [(ip, xor_distance(my_ip, ip)) for ip in all_ips if ip != my_ip]
    candidates.sort(key=lambda t: t[1])
    neighbors = [ip for ip, _ in candidates[:k]]
    logging.debug(
        f"[Topology] {my_ip} → voisins : {neighbors} "
        f"(sur {len(all_ips)} nœuds connus)"
    )
    return neighbors


def build_routing_table(my_ip: str,
                        all_ips: List[str],
                        k: int) -> Dict:
    """Retourne la table de routage complète (utile pour débogage/stats)."""
    neighbors = get_k_nearest_neighbors(my_ip, all_ips, k)
    return {
        "my_ip":     my_ip,
        "my_hash":   hash_ip(my_ip),
        "k":         k,
        "neighbors": [
            {
                "ip":       ip,
                "hash":     hash_ip(ip),
                "distance": xor_distance(my_ip, ip),
            }
            for ip in neighbors
        ],
    }
