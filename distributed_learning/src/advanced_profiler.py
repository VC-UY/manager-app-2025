"""
AdvancedProfiler — profilage fin par lecture directe de /proc
==============================================================

Objectif
--------
Distinguer un BUG LOGICIEL (ex. fuite mémoire, modèle trop gros) d'une
CONTRAINTE D'ENVIRONNEMENT (ex. throttling thermique, RAM partagée saturée)
en collectant des métriques fiables avant/pendant/après l'entraînement.

Métriques collectées et signification
-------------------------------------
- RSS_baseline   : empreinte RAM avant tout chargement -> sépare le runtime du payload.
- RSS_peak       : pic absolu de RAM physique -> tient-il sur le nœud le plus contraint ?
- RSS_delta      : peak - baseline -> coût mémoire RÉEL de la tâche.
- RSS_avg        : pression moyenne -> ordonnancement multi-tâches.
- RSS_profile    : courbe temporelle -> détecte pics, fuites, libérations.
- PSS / USS      : PSS = RSS partagée proportionnellement ; USS = mémoire UNIQUE
                   (écarte les libs partagées, vrai coût "marginal").
- CPU_avg        : intensité CPU moyenne -> CPU-bound vs IO-bound.
- Throttle_ratio : 1 - (freq_cur_moy / freq_max). > 10% = résultats non reproductibles.
- ETE            : durée wall-clock (End-to-End) de la phase mesurée.
- IPC            : Instructions Par Cycle (perf stat). Distingue un CPU lent
                   (throttle, IPC normal) d'un code inefficace (IPC bas, cache miss).

Sources
-------
- /proc/[pid]/status                          -> VmRSS, VmPeak
- /proc/[pid]/smaps_rollup                    -> Pss, Private_*, Shared_*
- /sys/devices/system/cpu/cpu*/cpufreq/...    -> fréquence courante et max
- subprocess: perf stat -e instructions,cycles -> IPC
- Fallback psutil pour systèmes non-Linux
"""
import logging
import os
import platform
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None


class AdvancedProfiler:
    """Profileur avancé lisant /proc et /sys pour des métriques système fiables."""

    def __init__(self,
                 pid: Optional[int] = None,
                 sample_interval: float = 0.5,
                 max_samples: int = 300):
        """
        Args:
            pid             : PID à profiler (défaut : processus courant).
            sample_interval : période d'échantillonnage en secondes.
            max_samples     : nombre max d'échantillons conservés (downsampling au-delà).
        """
        self.pid = pid or os.getpid()
        self.sample_interval = float(sample_interval)
        self.max_samples = int(max_samples)
        self.is_linux = platform.system() == "Linux"

        # Baseline
        self.baseline: Dict[str, float] = {}
        self.cpu_max_mhz: float = 0.0
        self.start_ts: float = 0.0
        self.stop_ts: float = 0.0
        self.start_ts_mono: float = 0.0
        self.stop_ts_mono: float = 0.0

        # Echantillons (timestamp, rss_kb, pss_kb, uss_kb, cpu_pct, freq_mhz)
        self._samples: List[Tuple[float, int, int, int, float, float]] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Résultats
        self._ipc: Optional[float] = None
        self._method_notes: Dict[str, str] = {}

        if psutil is not None:
            try:
                self._ps_proc = psutil.Process(self.pid)
                self._ps_proc.cpu_percent(interval=None)   # init du compteur
            except Exception:
                self._ps_proc = None
        else:
            self._ps_proc = None

    # ─── Lecture /proc et /sys ───────────────────────────────────────────────
    @staticmethod
    def _read_proc_status(pid: int) -> Dict[str, int]:
        """Lit /proc/[pid]/status -> dict {VmRSS, VmPeak, Threads, ...} en kB."""
        path = f"/proc/{pid}/status"
        result: Dict[str, int] = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    key, val = line.split(":", 1)
                    val = val.strip()
                    # Lignes du style "VmRSS:    12345 kB"
                    m = re.match(r"^(\d+)(?:\s*kB)?$", val)
                    if m:
                        result[key.strip()] = int(m.group(1))
        except FileNotFoundError:
            pass
        return result

    @staticmethod
    def _read_smaps_rollup(pid: int) -> Dict[str, int]:
        """Lit /proc/[pid]/smaps_rollup -> dict {Rss, Pss, Private_*, Shared_*} en kB."""
        path = f"/proc/{pid}/smaps_rollup"
        result: Dict[str, int] = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    key, val = line.split(":", 1)
                    val = val.strip()
                    m = re.match(r"^(\d+)(?:\s*kB)?$", val)
                    if m:
                        result[key.strip()] = int(m.group(1))
        except FileNotFoundError:
            pass
        return result

    @staticmethod
    def _read_cpu_freqs() -> List[float]:
        """Liste des fréquences courantes (MHz) par cœur via /sys."""
        freqs: List[float] = []
        try:
            cpu_dirs = sorted(
                d for d in os.listdir("/sys/devices/system/cpu/")
                if re.match(r"^cpu\d+$", d)
            )
            for cpu in cpu_dirs:
                p = f"/sys/devices/system/cpu/{cpu}/cpufreq/scaling_cur_freq"
                try:
                    with open(p, "r") as f:
                        khz = int(f.read().strip())
                        freqs.append(khz / 1000.0)
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            pass

        # Fallback psutil
        if not freqs and psutil is not None:
            try:
                pf = psutil.cpu_freq(percpu=True)
                if pf:
                    freqs = [c.current for c in pf]
                else:
                    cf = psutil.cpu_freq()
                    if cf:
                        freqs = [cf.current]
            except Exception:
                pass
        return freqs

    @staticmethod
    def _read_cpu_max_freq() -> float:
        """Fréquence max théorique (MHz). 0.0 si inconnu."""
        max_vals: List[float] = []
        try:
            cpu_dirs = sorted(
                d for d in os.listdir("/sys/devices/system/cpu/")
                if re.match(r"^cpu\d+$", d)
            )
            for cpu in cpu_dirs:
                p = f"/sys/devices/system/cpu/{cpu}/cpufreq/cpuinfo_max_freq"
                try:
                    with open(p, "r") as f:
                        max_vals.append(int(f.read().strip()) / 1000.0)
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            pass

        if max_vals:
            return max(max_vals)

        # Fallback psutil
        if psutil is not None:
            try:
                cf = psutil.cpu_freq()
                if cf and cf.max:
                    return cf.max
            except Exception:
                pass
        return 0.0

    # ─── Baseline ────────────────────────────────────────────────────────────
    def capture_baseline(self) -> Dict[str, float]:
        """
        Capture l'empreinte mémoire AVANT le chargement des données / modèle.
        À appeler le plus tôt possible dans le cycle de vie du volontaire.
        """
        rss_baseline = 0
        pss_baseline = 0
        uss_baseline = 0

        status = self._read_proc_status(self.pid)
        if "VmRSS" in status:
            rss_baseline = status["VmRSS"]
        elif self._ps_proc is not None:
            try:
                rss_baseline = self._ps_proc.memory_info().rss // 1024
            except Exception:
                pass

        smaps = self._read_smaps_rollup(self.pid)
        if smaps:
            pss_baseline = smaps.get("Pss", 0)
            uss_baseline = (smaps.get("Private_Clean", 0)
                            + smaps.get("Private_Dirty", 0))
            self._method_notes["smaps_rollup"] = "OK (Linux)"
        else:
            self._method_notes["smaps_rollup"] = "Indisponible (non-Linux ou kernel ancien)"

        self.cpu_max_mhz = self._read_cpu_max_freq()
        if self.cpu_max_mhz == 0.0:
            self._method_notes["cpu_max_freq"] = "Inconnue -> Throttle_ratio non calculable"
        else:
            self._method_notes["cpu_max_freq"] = f"{self.cpu_max_mhz:.0f} MHz"

        self.baseline = {
            "rss_baseline_kb": rss_baseline,
            "pss_baseline_kb": pss_baseline,
            "uss_baseline_kb": uss_baseline,
            "ts": time.time(),
        }
        logging.info(
            f"[AdvProfiler] Baseline capturée : "
            f"RSS={rss_baseline} kB, PSS={pss_baseline} kB, USS={uss_baseline} kB, "
            f"CPU_max={self.cpu_max_mhz:.0f} MHz"
        )
        return self.baseline

    # ─── Monitoring continu ──────────────────────────────────────────────────
    def start_monitoring(self):
        """Démarre le thread d'échantillonnage périodique."""
        with self._lock:
            self._samples.clear()
            self._ipc = None
            self.start_ts = time.time()
            self.start_ts_mono = time.monotonic()
            self.stop_ts = 0.0
            self.stop_ts_mono = 0.0
            self._monitoring = True

        def _loop():
            # Premier appel cpu_percent pour initialiser
            if self._ps_proc is not None:
                try:
                    self._ps_proc.cpu_percent(interval=None)
                except Exception:
                    pass

            while self._monitoring:
                try:
                    ts = time.monotonic()
                    status = self._read_proc_status(self.pid)
                    rss = status.get("VmRSS", 0)
                    if rss == 0 and self._ps_proc is not None:
                        try:
                            rss = self._ps_proc.memory_info().rss // 1024
                        except Exception:
                            rss = 0

                    smaps = self._read_smaps_rollup(self.pid)
                    pss = smaps.get("Pss", 0)
                    uss = (smaps.get("Private_Clean", 0)
                           + smaps.get("Private_Dirty", 0))

                    cpu_pct = 0.0
                    if self._ps_proc is not None:
                        try:
                            cpu_pct = self._ps_proc.cpu_percent(interval=None)
                        except Exception:
                            pass

                    freqs = self._read_cpu_freqs()
                    freq_mhz = sum(freqs) / len(freqs) if freqs else 0.0

                    with self._lock:
                        self._samples.append((ts, rss, pss, uss, cpu_pct, freq_mhz))
                        # Downsampling si dépassement
                        if len(self._samples) > self.max_samples:
                            # On garde 1 échantillon sur 2 (uniforme)
                            self._samples = self._samples[::2]
                except Exception as e:
                    logging.debug(f"[AdvProfiler] Erreur sampling : {e}")
                time.sleep(self.sample_interval)

        self._monitor_thread = threading.Thread(target=_loop, daemon=True)
        self._monitor_thread.start()
        logging.debug("[AdvProfiler] Monitoring démarré.")

    def stop_monitoring(self) -> Dict[str, float]:
        """Arrête le monitoring et calcule toutes les métriques agrégées."""
        with self._lock:
            self._monitoring = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
        self.stop_ts = time.time()
        self.stop_ts_mono = time.monotonic()

        return self._compute_metrics()

    # ─── IPC via perf ────────────────────────────────────────────────────────
    def measure_ipc(self, duration_s: float = 2.0) -> Optional[float]:
        """
        Mesure best-effort de l'IPC via `perf stat`.
        Retourne None si perf indisponible ou erreur.

        ATTENTION : bloque pendant `duration_s` secondes.
        """
        try:
            cmd = ["perf", "stat", "-x,", "-e", "instructions,cycles",
                   "-p", str(self.pid), "sleep", str(duration_s)]
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=duration_s + 3.0,
            )
            # perf écrit ses stats sur stderr
            instructions = 0
            cycles = 0
            for line in proc.stderr.split("\n"):
                parts = line.split(",")
                if len(parts) >= 3:
                    val_str = parts[0].strip()
                    metric = parts[2].strip()
                    if metric == "instructions":
                        try:
                            instructions = int(val_str.replace(" ", ""))
                        except ValueError:
                            pass
                    elif metric == "cycles":
                        try:
                            cycles = int(val_str.replace(" ", ""))
                        except ValueError:
                            pass
            if cycles > 0 and instructions > 0:
                self._ipc = instructions / cycles
                self._method_notes["ipc"] = f"OK via perf ({duration_s}s)"
                return self._ipc
            self._method_notes["ipc"] = "perf a tourné mais aucune valeur exploitable"
        except FileNotFoundError:
            self._method_notes["ipc"] = "perf non installé"
        except subprocess.TimeoutExpired:
            self._method_notes["ipc"] = "perf timeout"
        except Exception as e:
            self._method_notes["ipc"] = f"erreur perf : {e}"
        return None

    # ─── Agrégation des métriques ────────────────────────────────────────────
    def _compute_metrics(self) -> Dict[str, float]:
        with self._lock:
            samples = list(self._samples)

        if not samples:
            logging.warning("[AdvProfiler] Aucun échantillon collecté.")
            return {}

        rss_vals = [s[1] for s in samples]
        pss_vals = [s[2] for s in samples]
        uss_vals = [s[3] for s in samples]
        cpu_vals = [s[4] for s in samples]
        freq_vals = [s[5] for s in samples if s[5] > 0]

        rss_baseline = self.baseline.get("rss_baseline_kb", 0)

        rss_peak = max(rss_vals)
        rss_avg = sum(rss_vals) / len(rss_vals)
        rss_delta = rss_peak - rss_baseline

        cpu_avg = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0

        # Throttle
        if freq_vals and self.cpu_max_mhz > 0:
            freq_avg = sum(freq_vals) / len(freq_vals)
            throttle_ratio = max(0.0, 1.0 - (freq_avg / self.cpu_max_mhz))
        else:
            freq_avg = 0.0
            throttle_ratio = -1.0   # sentinelle : non mesurable

        ete = self.stop_ts_mono - self.start_ts_mono if self.stop_ts_mono > 0 else 0.0

        # Downsampling profil (≤ 100 points pour JSON)
        ts_origin = samples[0][0]
        step = max(1, len(samples) // 100)
        profile = [
            (round(s[0] - ts_origin, 3), s[1])
            for s in samples[::step]
        ]

        return {
            # Mémoire
            "rss_baseline_kb": rss_baseline,
            "rss_peak_kb": rss_peak,
            "rss_avg_kb": round(rss_avg, 1),
            "rss_delta_kb": rss_delta,
            "pss_peak_kb": max(pss_vals) if any(pss_vals) else 0,
            "pss_avg_kb": round(sum(pss_vals) / len(pss_vals), 1) if pss_vals else 0,
            "uss_peak_kb": max(uss_vals) if any(uss_vals) else 0,
            "uss_avg_kb": round(sum(uss_vals) / len(uss_vals), 1) if uss_vals else 0,
            "rss_profile": profile,   # liste de (t_relatif_s, rss_kb)
            # CPU
            "cpu_avg_pct": round(cpu_avg, 2),
            "cpu_max_mhz": self.cpu_max_mhz,
            "cpu_avg_freq_mhz": round(freq_avg, 1),
            "throttle_ratio": round(throttle_ratio, 4) if throttle_ratio >= 0 else None,
            # Temps
            "ete_seconds": round(ete, 3),
            "n_samples": len(samples),
            # IPC
            "ipc": round(self._ipc, 4) if self._ipc is not None else None,
        }

    def get_full_report(self) -> Dict:
        """Retourne le rapport complet (métriques + notes méthodologiques)."""
        metrics = self._compute_metrics()
        return {
            "pid": self.pid,
            "platform": platform.platform(),
            "metrics": metrics,
            "method_notes": self._method_notes,
        }

    # ─── Context manager ─────────────────────────────────────────────────────
    def __enter__(self):
        if not self.baseline:
            self.capture_baseline()
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()
