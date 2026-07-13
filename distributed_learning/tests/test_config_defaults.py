import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def test_gossip_interval_is_not_allowed_to_block_experiments(monkeypatch):
    monkeypatch.setenv("GOSSIP_INTERVAL", "5000")
    monkeypatch.delenv("HEARTBEAT_INTERVAL", raising=False)
    monkeypatch.delenv("HEARTBEAT_TIMEOUT", raising=False)
    monkeypatch.delenv("SOCKET_TIMEOUT", raising=False)
    monkeypatch.delenv("MAX_ROUNDS", raising=False)
    monkeypatch.delenv("LOCAL_EPOCHS", raising=False)
    monkeypatch.delenv("BATCH_SIZE", raising=False)
    monkeypatch.delenv("LEARNING_RATE", raising=False)

    reloaded = importlib.reload(config)

    assert reloaded.GOSSIP_INTERVAL <= 600
    assert reloaded.GOSSIP_INTERVAL >= 1
