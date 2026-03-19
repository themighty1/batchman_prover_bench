"""
PCS server client — sends segment data for subset membership proofs.

Talks to the WHIR PCS server at localhost:9477.
Protocol: length-prefixed binary over TCP.
  [4B LE payload_length][1B msg_type][payload]
"""

import socket
import struct
import subprocess
import os

MSG_ADD_Z_ROOTS = 0x01
MSG_ADD_Q_ROOTS = 0x02
MSG_PROVE       = 0x03
MSG_CONFIG      = 0x04
MSG_OK          = 0x10
MSG_PROOF       = 0x11
MSG_ERROR       = 0xFF

PCS_HOST = '127.0.0.1'
PCS_PORT = 9477


class PCSClient:
    def __init__(self, host=PCS_HOST, port=PCS_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def close(self):
        self.sock.close()

    def _send_msg(self, msg_type: int, payload: bytes):
        header = struct.pack('<IB', len(payload) + 1, msg_type)
        self.sock.sendall(header + payload)

    def _recv_msg(self):
        """Receive a length-prefixed message. Returns (msg_type, payload)."""
        raw_len = self._recv_exact(4)
        total_len = struct.unpack('<I', raw_len)[0]
        data = self._recv_exact(total_len)
        msg_type = data[0]
        payload = data[1:]
        return msg_type, payload

    def _recv_exact(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed")
            buf += chunk
        return buf

    def send_config(self, config: dict):
        import json
        payload = json.dumps(config).encode()
        self._send_msg(MSG_CONFIG, payload)
        msg_type, _ = self._recv_msg()
        assert msg_type == MSG_OK, f"Expected OK, got {msg_type:#x}"

    def send_z_roots(self, roots: list[int]):
        """Send active MACs as Z roots (u64 LE each)."""
        payload = b''.join(struct.pack('<Q', r) for r in roots)
        self._send_msg(MSG_ADD_Z_ROOTS, payload)
        msg_type, _ = self._recv_msg()
        assert msg_type == MSG_OK, f"Expected OK, got {msg_type:#x}"

    def send_q_roots(self, roots: list[int]):
        """Send all keys as Q roots (u64 LE each)."""
        payload = b''.join(struct.pack('<Q', r) for r in roots)
        self._send_msg(MSG_ADD_Q_ROOTS, payload)
        msg_type, _ = self._recv_msg()
        assert msg_type == MSG_OK, f"Expected OK, got {msg_type:#x}"

    def prove(self, alpha: int):
        """Request proof generation. Returns proof bytes."""
        payload = struct.pack('<Q', alpha)
        self._send_msg(MSG_PROVE, payload)
        msg_type, payload = self._recv_msg()
        if msg_type == MSG_ERROR:
            raise RuntimeError(f"PCS server error: {payload.decode()}")
        assert msg_type == MSG_PROOF, f"Expected PROOF, got {msg_type:#x}"
        return payload


def compute_segment_keys(segment_dir: str, segment_keys_bin: str) -> tuple[list[int], list[int]]:
    """
    Run segment_keys binary on a segment directory.
    Returns (active_macs, all_keys) as lists of u64.
    """
    result = subprocess.run(
        [segment_keys_bin, segment_dir],
        capture_output=True, check=True)

    data = result.stdout
    num_active, num_all = struct.unpack_from('<II', data, 0)
    offset = 8

    active_macs = []
    for i in range(num_active):
        v, = struct.unpack_from('<Q', data, offset)
        active_macs.append(v)
        offset += 8

    all_keys = []
    for i in range(num_all):
        v, = struct.unpack_from('<Q', data, offset)
        all_keys.append(v)
        offset += 8

    return active_macs, all_keys
