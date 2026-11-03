import base64
import hashlib
import hmac
import io
import json

import numpy as np

MARKER = "AEGIS_REPORT "


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sign(key: bytes, nonce: str, report: dict) -> str:
    return hmac.new(key, (nonce + canon(report)).encode(), hashlib.sha256).hexdigest()


def blob_digest(blobs: dict) -> str:
    h = hashlib.sha256()
    for f in blobs.get("frames", []):
        h.update(f.encode())
    for k in sorted(blobs.get("keyframes", {})):
        h.update(k.encode())
        h.update(canon(blobs["keyframes"][k]).encode())
    return h.hexdigest()


def verify(key: bytes, nonce: str, envelope: dict):
    report = envelope.get("report") or {}
    sig = envelope.get("sig", "")
    if report.get("nonce") != nonce:
        return False, "nonce mismatch"
    if not hmac.compare_digest(sign(key, nonce, report), sig):
        return False, "bad HMAC signature"
    if report.get("blob_digest") != blob_digest(envelope.get("blobs", {})):
        return False, "frame payload was tampered with"
    return True, "nonce + HMAC verified"


def pack_array_bundle(**arrays) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode()


def unpack_array_bundle(b64: str) -> dict:
    data = np.load(io.BytesIO(base64.b64decode(b64)))
    return {k: data[k] for k in data.files}


def parse_stdout(text: str):
    env = None
    for line in text.splitlines():
        if line.startswith(MARKER):
            try:
                env = json.loads(line[len(MARKER):])
            except json.JSONDecodeError:
                continue
    return env
