import logging
from pathlib import Path

from aegis.config import settings
from aegis.models.meta_client import MetaClient

log = logging.getLogger("aegis.voice")

MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".webm": "audio/webm"}


def _transcript(data):
    if isinstance(data, str):
        return data.strip()
    if not isinstance(data, dict):
        return ""
    for key in ("text", "transcript"):
        if isinstance(data.get(key), str) and data[key].strip():
            return data[key].strip()
    for key in ("turns", "segments", "utterances", "results"):
        items = data.get(key)
        if isinstance(items, list) and items:
            parts = [(i.get("text") or i.get("transcript") or "").strip() for i in items if isinstance(i, dict)]
            joined = " ".join(p for p in parts if p)
            if joined:
                return joined
    return ""


class VoiceTranscriber:
    name = f"Muse Voice Transcribe ({settings.voice_model})"

    def __init__(self):
        self.client = MetaClient()

    def transcribe(self, audio: bytes, filename="command.wav"):
        mime = MIME.get(Path(filename).suffix.lower(), "application/octet-stream")
        data = self.client.post(settings.voice_endpoint, files={"file": (filename, audio, mime)},
                                data={"model": settings.voice_model})
        text = _transcript(data)
        if not text:
            raise RuntimeError(f"no transcript in response keys {list(data)[:8] if isinstance(data, dict) else data!r}")
        return text


def transcribe(audio: bytes, filename="command.wav"):
    # returns (text, backend). Voice has no offline twin: without a key the caller keeps the typed instruction.
    if settings.backend("voice") != "live":
        raise RuntimeError("voice needs MODEL_API_KEY (Muse Voice Transcribe has no offline stand-in)")
    t = VoiceTranscriber()
    return t.transcribe(audio, filename), t.name
