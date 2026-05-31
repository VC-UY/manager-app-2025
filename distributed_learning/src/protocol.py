"""
Protocole réseau : encodage/décodage des messages.

Format : [4 octets taille header][header JSON UTF-8][payload binaire]
"""
import json
import struct
import socket
import logging
from typing import Tuple

# ─── Types de messages ────────────────────────────────────────────────────────
MSG_HEARTBEAT          = "heartbeat"
MSG_VOLUNTEER_LIST     = "volunteer_list"
MSG_SEND_MODEL         = "send_model"
MSG_POLL_MODELS        = "poll_models"
MSG_MODEL_DELIVERY     = "model_delivery"
MSG_REQUEST_NEIGHBORS  = "request_neighbors"
MSG_NEIGHBORS_RESPONSE = "neighbors_response"
MSG_STATS_REQUEST      = "stats_request"
MSG_STATS_RESPONSE     = "stats_response"
MSG_STATS_PUSH         = "stats_push"
MSG_ACK                = "ack"
MSG_ERROR              = "error"
MSG_DISCONNECT         = "disconnect"

_HDR_FMT   = ">I"          # unsigned int 32-bit big-endian
_HDR_BYTES = struct.calcsize(_HDR_FMT)
_MAX_HEADER = 1 * 1024 * 1024   # 1 MB
_MAX_PAYLOAD = 500 * 1024 * 1024  # 500 MB


def send_message(sock: socket.socket,
                 msg_type: str,
                 data: dict,
                 payload: bytes = b"") -> None:
    """Encode et envoie un message sur le socket."""
    header_dict = {"type": msg_type, "data": data, "payload_size": len(payload)}
    header_bytes = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    frame = struct.pack(_HDR_FMT, len(header_bytes)) + header_bytes + payload
    # sendall gère les envois fragmentés
    sock.sendall(frame)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Lit exactement n octets depuis le socket, lève ConnectionError si fermé."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Connexion fermée par le pair")
        buf.extend(chunk)
    return bytes(buf)


def receive_message(sock: socket.socket) -> Tuple[str, dict, bytes]:
    """
    Reçoit et décode un message depuis le socket.

    Retourne (msg_type, data_dict, payload_bytes).
    """
    # Taille du header
    raw_size = _recv_exactly(sock, _HDR_BYTES)
    hdr_size = struct.unpack(_HDR_FMT, raw_size)[0]
    if hdr_size > _MAX_HEADER:
        raise ValueError(f"Header trop grand : {hdr_size} octets")

    # Header JSON
    hdr_bytes = _recv_exactly(sock, hdr_size)
    hdr = json.loads(hdr_bytes.decode("utf-8"))

    # Payload binaire
    payload_size = hdr.get("payload_size", 0)
    if payload_size > _MAX_PAYLOAD:
        raise ValueError(f"Payload trop grand : {payload_size} octets")
    payload = _recv_exactly(sock, payload_size) if payload_size > 0 else b""

    return hdr["type"], hdr.get("data", {}), payload
