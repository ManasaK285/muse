import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv():
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name, default=""):
    return os.environ.get(name, default)


@dataclass
class Settings:
    api_key: str = _env("MODEL_API_KEY") or _env("META_API_KEY")
    base_url: str = _env("META_API_BASE", "https://api.meta.ai/v1").rstrip("/")
    spark_model: str = _env("MUSE_SPARK_MODEL", "muse-spark-1.2")
    code_model: str = _env("MUSE_CODE_MODEL", "muse-spark-1.2")
    code_cmd: str = _env("MUSE_CODE_CMD")  # optional: shell out to the Muse Code CLI instead
    wonder_model: str = _env("MUSE_WONDER_MODEL", "muse-wonder-1.0")
    image_model: str = _env("MUSE_IMAGE_MODEL", "muse-image-1.0")
    image_url: str = _env("MUSE_IMAGE_URL")
    sam_model: str = _env("SAM_MODEL", "sam-3.1")
    sam_url: str = _env("SAM_URL")
    sam_max_carry: int = int(_env("AEGIS_SAM_MAX_CARRY", "3"))
    voice_model: str = _env("MUSE_VOICE_MODEL", "muse-voice-transcribe-1.0")
    voice_url: str = _env("MUSE_VOICE_URL")
    # Muse Spark Open runs on a hosted provider (Fireworks, Baseten, ...) behind an OpenAI-compatible API
    spark_open_base: str = _env("SPARK_OPEN_BASE_URL").rstrip("/")
    spark_open_key: str = _env("SPARK_OPEN_API_KEY")
    spark_open_model: str = _env("SPARK_OPEN_MODEL")
    # Muse Glimmer runs on your own machine (llama.cpp, vLLM, SGLang, Ollama ...)
    glimmer_base: str = _env("GLIMMER_BASE_URL").rstrip("/")
    glimmer_key: str = _env("GLIMMER_API_KEY")
    glimmer_model: str = _env("GLIMMER_MODEL", "muse-glimmer")
    redteam_n: int = int(_env("AEGIS_REDTEAM_N", "3"))
    reasoning_effort: str = _env("MUSE_REASONING_EFFORT", "medium")
    sandbox: str = _env("AEGIS_SANDBOX", "local")
    max_iters: int = int(_env("AEGIS_MAX_ITERS", "4"))
    timeout_s: float = float(_env("AEGIS_HTTP_TIMEOUT", "120"))

    def backend(self, component: str) -> str:
        # AEGIS_<COMPONENT>_BACKEND=live|offline wins, then AEGIS_MODE, then key presence
        specific = _env(f"AEGIS_{component.upper()}_BACKEND")
        if specific:
            return specific.lower()
        mode = _env("AEGIS_MODE").lower()
        if mode in ("live", "offline"):
            return mode
        return "live" if self.api_key else "offline"

    def disabled(self, component: str) -> bool:
        # for models that do not use the Meta key (hosted Spark Open, local Glimmer): only an explicit offline switch turns them off
        specific = _env(f"AEGIS_{component.upper()}_BACKEND").lower()
        return specific == "offline" or (not specific and _env("AEGIS_MODE").lower() == "offline")

    @property
    def sam_endpoint(self):
        return self.sam_url or f"{self.base_url}/responses"

    @property
    def voice_endpoint(self):
        return self.voice_url or f"{self.base_url}/asr/transcribe"

    @property
    def image_endpoint(self):
        return self.image_url or f"{self.base_url}/images/edits"


settings = Settings()
