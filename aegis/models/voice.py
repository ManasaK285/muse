import io
import json
import logging
import wave
from pathlib import Path

from aegis.config import settings
from aegis.models.meta_client import MetaAPIError, MetaClient

log = logging.getLogger("aegis.voice")

MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".webm": "audio/webm"}

# names the same 16-bit PCM WAV goes by across speech APIs; the first one Meta accepts wins
ENCODINGS = ["LINEAR16", "WAV", "PCM_S16LE", "pcm_s16le", "wav", "linear16", "pcm16"]
RATE_KEYS = ("sampleRate", "sampleRateHertz", "sample_rate", "sample_rate_hertz")
CHANNEL_KEYS = ("channels", "audioChannelCount", "numChannels", "channelCount")


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


def _wav_info(audio):
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return w.getframerate(), w.getnchannels()
    except Exception:
        return 16000, 1


class VoiceTranscriber:
    name = f"Muse Voice Transcribe ({settings.voice_model})"

    def __init__(self):
        self.client = MetaClient()
        self.attempts = []

    def _send(self, request, filename, audio, mime):
        r = self.client.http.post(settings.voice_endpoint, files={
            "request": ("request.json", json.dumps(request).encode(), "application/json"),
            "audio": (filename, audio, mime),
        })
        try:
            body = r.json()
        except ValueError:
            body = {"raw": r.text[:300]}
        return r.status_code, body

    def transcribe(self, audio: bytes, filename="command.wav"):
        mime = MIME.get(Path(filename).suffix.lower(), "application/octet-stream")
        rate, channels = _wav_info(audio)
        request = {"model": settings.voice_model}
        enc = iter(ENCODINGS)
        request["audioEncoding"] = next(enc)
        for _ in range(12):
            status, body = self._send(request, filename, audio, mime)
            err = body.get("error") if isinstance(body, dict) else None
            self.attempts.append({"request": dict(request), "status": status,
                                  "message": (err or {}).get("message") if isinstance(err, dict) else None})
            if status < 400:
                text = _transcript(body)
                if not text:
                    raise RuntimeError(f"no transcript in response keys {list(body)[:8]}")
                log.info("voice accepted request %s", request)
                return text
            if status != 400 or not isinstance(err, dict):
                raise MetaAPIError(f"{settings.voice_endpoint} -> {status}: {json.dumps(body)[:400]}")
            param = err.get("param") or ""
            # the API names the field it objects to, so adjust exactly that field and retry
            if param == "audioEncoding":
                try:
                    request["audioEncoding"] = next(enc)
                    continue
                except StopIteration:
                    break
            if param in RATE_KEYS:
                request[param] = rate
                continue
            if param in CHANNEL_KEYS:
                request[param] = channels
                continue
            break
        tried = "; ".join(f"{a['request']} -> {a['status']} {a['message']}" for a in self.attempts[-4:])
        raise MetaAPIError(f"voice request not accepted. Last attempts: {tried}")


def transcribe(audio: bytes, filename="command.wav"):
    # returns (text, backend). Voice has no offline twin: without a key the caller keeps the typed instruction.
    if settings.backend("voice") != "live":
        raise RuntimeError("voice needs MODEL_API_KEY (Muse Voice Transcribe has no offline stand-in)")
    t = VoiceTranscriber()
    return t.transcribe(audio, filename), t.name
