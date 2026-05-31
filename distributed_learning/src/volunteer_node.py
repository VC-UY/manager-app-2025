"""
Modèle de données pour représenter un nœud volontaire.

Chaque volontaire est identifié de manière unique par son adresse MAC (immuable),
et fournit des informations sur ses ressources allouées au système.
"""
import json
import socket
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import subprocess
import psutil


@dataclass
class ResourceInfo:
    """Informations sur les ressources d'un volontaire."""
    cpu_cores: int                    # Nombre de cœurs CPU alloués
    cpu_freq_ghz: float              # Fréquence CPU en GHz
    ram_gb: float                    # RAM allouée en GB
    network_bandwidth_mbps: float    # Bande passante réseau en Mbps
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceInfo":
        return cls(**data)


@dataclass
class VolunteerNode:
    """
    Représente un nœud volontaire dans le réseau.
    
    Identifié uniquement par son MAC address (immuable).
    L'IP peut changer (DHCP, NAT), donc elle est séparée.
    """
    mac_address: str              # Identifiant unique (ex: "AA:BB:CC:DD:EE:FF")
    resources: ResourceInfo        # Ressources allouées
    current_ip: Optional[str] = None  # IP actuelle (peut changer)
    last_heartbeat: float = field(default_factory=lambda: 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le nœud pour transmission réseau."""
        return {
            "mac_address": self.mac_address,
            "current_ip": self.current_ip,
            "resources": self.resources.to_dict(),
            "last_heartbeat": self.last_heartbeat,
        }
    
    def to_json(self) -> str:
        """Convertit en JSON."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VolunteerNode":
        """Désérialise depuis un dict."""
        return cls(
            mac_address=data["mac_address"],
            resources=ResourceInfo.from_dict(data["resources"]),
            current_ip=data.get("current_ip"),
            last_heartbeat=data.get("last_heartbeat", 0.0),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "VolunteerNode":
        """Désérialise depuis JSON."""
        return cls.from_dict(json.loads(json_str))


# ─── Utilitaires de détection système ──────────────────────────────────────────


def get_mac_address(remote_host: str = "8.8.8.8") -> str:
    """
    Obtient l'adresse MAC de la carte réseau utilisée pour atteindre remote_host.

    Stratégies (par ordre de priorité) :
      1. Associer l'IP de sortie à une interface via psutil.net_if_addrs(),
         puis lire son adresse MAC via la famille AF_LINK.
      2. Lire /sys/class/net/<iface>/address — stable avec network_mode: host.
      3. Parser la sortie de `ip link show`.
      4. Fallback : pseudo-MAC déterministe basée sur le hostname.

    En cas d'échec, retourne une adresse MAC fictive basée sur le hostname.
    """
    # ── Stratégie 1 : IP de sortie → interface → MAC via psutil AF_LINK ──────
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((remote_host, 80))
        outbound_ip = s.getsockname()[0]
        s.close()

        for iface_name, addrs in psutil.net_if_addrs().items():
            # Cette interface porte-t-elle l'IP de sortie ?
            if not any(a.family == socket.AF_INET and a.address == outbound_ip
                       for a in addrs):
                continue
            # Chercher l'adresse MAC (AF_LINK) parmi les adresses de l'interface
            for a in addrs:
                if a.family == psutil.AF_LINK and a.address:
                    return a.address.upper()
    except Exception:
        pass

    # ── Stratégie 2 : /sys/class/net (fiable sous Linux, y compris Docker) ───
    try:
        from pathlib import Path
        for p in sorted(Path("/sys/class/net").iterdir()):
            # Ignorer les interfaces virtuelles Docker et loopback
            if p.name.startswith(("docker", "veth", "br-", "lo")):
                continue
            mac = (p / "address").read_text().strip()
            if mac and mac != "00:00:00:00:00:00":
                return mac.upper()
    except Exception:
        pass

    # ── Stratégie 3 : parser `ip link show` ──────────────────────────────────
    try:
        result = subprocess.run(
            ["ip", "link", "show"],
            capture_output=True, text=True, timeout=2,
        )
        for line in result.stdout.split("\n"):
            if "link/ether" in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    return parts[1].upper()
    except Exception:
        pass

    # ── Stratégie 4 : pseudo-MAC déterministe basée sur le hostname ──────────
    import hashlib
    hostname = socket.gethostname()
    hash_obj = hashlib.md5(hostname.encode())
    hash_hex = hash_obj.hexdigest()[:12]
    mac = ":".join(hash_hex[i : i + 2] for i in range(0, 12, 2))
    return mac.upper()


def get_resource_info(
    cpu_cores: Optional[int] = None,
    cpu_freq_ghz: Optional[float] = None,
    ram_gb: Optional[float] = None,
    network_bandwidth_mbps: Optional[float] = None
) -> ResourceInfo:
    """
    Obtient les informations de ressources du système.
    
    Les paramètres optionnels permettent de surcharger les valeurs détectées automatiquement.
    Utile pour spécifier des ressources allouées vs disponibles.
    """
    # CPU cores
    if cpu_cores is None:
        cpu_cores = psutil.cpu_count(logical=True) or 1
    
    # CPU frequency
    if cpu_freq_ghz is None:
        try:
            freq = psutil.cpu_freq()
            cpu_freq_ghz = freq.max / 1000.0 if freq else 2.0
        except:
            cpu_freq_ghz = 2.0
    
    # RAM total
    if ram_gb is None:
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024**3)
    
    # Network bandwidth (détection difficile, valeur par défaut 1000 Mbps)
    if network_bandwidth_mbps is None:
        network_bandwidth_mbps = 1000.0
    
    return ResourceInfo(
        cpu_cores=cpu_cores,
        cpu_freq_ghz=round(cpu_freq_ghz, 2),
        ram_gb=round(ram_gb, 2),
        network_bandwidth_mbps=round(network_bandwidth_mbps, 2),
    )
