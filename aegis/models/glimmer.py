import logging

from aegis.config import settings
from aegis.models.meta_client import CompatClient, extract_json

log = logging.getLogger("aegis.glimmer")


class EdgeModel:
    # Muse Glimmer served on this machine: the robot-side brain when the cloud API is unreachable
    def __init__(self):
        self.client = None
        self.last_error = None
        if settings.glimmer_base and not settings.disabled("glimmer"):
            try:
                self.client = CompatClient(settings.glimmer_base, settings.glimmer_key, settings.glimmer_model,
                                           timeout=180)
            except Exception as e:
                self.last_error = str(e)

    @property
    def available(self):
        return self.client is not None

    @property
    def name(self):
        return f"Muse Glimmer edge ({settings.glimmer_model} @ {settings.glimmer_base})"

    def ask_json(self, system, user, max_tokens=1500):
        txt = self.client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                               json_mode=True, max_tokens=max_tokens, temperature=0.2)
        return extract_json(txt)
