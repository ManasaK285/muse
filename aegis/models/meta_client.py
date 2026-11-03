import base64
import io
import json
import re
import time

import httpx
import numpy as np
from PIL import Image

from aegis.config import settings


class MetaAPIError(RuntimeError):
    pass


def png_data_url(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode_image(b64_or_url: str) -> np.ndarray:
    if b64_or_url.startswith("data:"):
        b64_or_url = b64_or_url.split(",", 1)[1]
    raw = base64.b64decode(b64_or_url)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


class MetaClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or settings.api_key
        self.base_url = (base_url or settings.base_url).rstrip("/")
        if not self.api_key:
            raise MetaAPIError("MODEL_API_KEY is not set")
        self.http = httpx.Client(timeout=settings.timeout_s, follow_redirects=True,
                                 headers={"Authorization": f"Bearer {self.api_key}"})

    def post(self, url, payload=None, retries=3, files=None, data=None):
        if not url.startswith("http"):
            url = self.base_url + url
        last = None
        for attempt in range(retries):
            try:
                if files is not None:
                    r = self.http.post(url, files=files, data=data)
                else:
                    r = self.http.post(url, json=payload)
            except httpx.HTTPError as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last = MetaAPIError(f"{r.status_code}: {r.text[:300]}")
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise MetaAPIError(f"{url} -> {r.status_code}: {r.text[:500]}")
            return r.json()
        raise MetaAPIError(f"{url} failed after {retries} tries: {last}")

    def stream_output_text(self, url, payload):
        # Responses API over SSE: collect response.output_text.delta until response.completed
        if not url.startswith("http"):
            url = self.base_url + url
        parts, status = [], "eof"
        with self.http.stream("POST", url, json={**payload, "stream": True},
                              headers={"Accept": "text/event-stream"}) as r:
            if r.status_code >= 400:
                r.read()
                raise MetaAPIError(f"{url} -> {r.status_code}: {r.text[:500]}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                try:
                    ev = json.loads(body)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("type", "")
                if kind == "response.output_text.delta":
                    parts.append(ev.get("delta", ""))
                elif kind == "response.completed":
                    status = "completed"
                    break
                elif kind == "response.incomplete":
                    status = "incomplete"
                    break
                elif kind in ("response.failed", "error"):
                    err = ev.get("error") or (ev.get("response") or {}).get("error") or ev
                    raise MetaAPIError(f"stream failed: {json.dumps(err)[:400]}")
        return "".join(parts), status

    def chat(self, model, messages, json_mode=False, max_tokens=4096):
        payload = {"model": model, "messages": messages, "max_completion_tokens": max_tokens,
                   "reasoning_effort": settings.reasoning_effort}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self.post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"] or ""


def fetch_image(url) -> np.ndarray:
    # signed URLs from the image endpoint are public, so a bare client is enough
    r = httpx.get(url, timeout=settings.timeout_s, follow_redirects=True)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def chat_json(client, model, system, user, max_tokens=4000):
    # one retry on a malformed or empty reply: reasoning models occasionally truncate or add prose
    messages = [{"role": "developer", "content": system}, {"role": "user", "content": user}]
    txt = client.chat(model, messages, json_mode=True, max_tokens=max_tokens)
    try:
        return extract_json(txt)
    except (ValueError, json.JSONDecodeError) as first:
        messages += [{"role": "assistant", "content": txt or "(empty)"},
                     {"role": "user", "content": f"That reply was not valid JSON ({first}). "
                                                 "Send the complete JSON object only, nothing else."}]
        txt = client.chat(model, messages, json_mode=True, max_tokens=int(max_tokens * 1.5))
        return extract_json(txt)


def extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        with_entry = [b for b in blocks if "def execute_policy" in b]
        return (with_entry or blocks)[-1].strip() + "\n"
    return text.strip() + "\n"


class CompatClient:
    # plain OpenAI-compatible chat, for hosted Spark Open and a local Glimmer server
    def __init__(self, base_url, api_key="", model="", timeout=None):
        if not base_url:
            raise MetaAPIError("no base URL configured")
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(timeout=timeout or settings.timeout_s, headers=headers)

    def chat(self, messages, json_mode=False, max_tokens=2048, temperature=0.4):
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = self.http.post(self.base_url + "/chat/completions", json=payload)
        if r.status_code == 400 and json_mode:
            payload.pop("response_format")  # some local servers reject it; the prompt still asks for JSON
            r = self.http.post(self.base_url + "/chat/completions", json=payload)
        if r.status_code >= 400:
            raise MetaAPIError(f"{self.base_url} -> {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"] or ""
