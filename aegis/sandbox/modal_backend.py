import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from aegis.sandbox import protocol
from aegis.sandbox.local import finish, make_job

APP_NAME = os.environ.get("AEGIS_MODAL_APP", "aegis-vla")


def build_image():
    import modal

    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential")
        .pip_install("pybullet==3.2.7", "numpy>=1.26", "pillow>=10")
        .add_local_python_source("aegis")
    )


class ModalSandbox:
    # one fresh sandbox per rollout; the HMAC key is minted here and only reaches the runner via stdin
    name = "modal-sandbox"

    def __init__(self, timeout=240, cpu=2.0):
        import modal

        self.modal = modal
        self.timeout = timeout
        self.cpu = cpu
        self.app = modal.App.lookup(APP_NAME, create_if_missing=True)
        self.image = build_image()

    def run(self, code, scenario, seed, enforce_gate=True, scene_graph=None, harness="aegis"):
        key, nonce, job = make_job(code, scenario, seed, enforce_gate, scene_graph, harness=harness)
        t0 = time.time()
        sb = self.modal.Sandbox.create(app=self.app, image=self.image, timeout=self.timeout,
                                       cpu=self.cpu, block_network=True)
        try:
            proc = sb.exec("python", "-m", "aegis.sandbox.runner", timeout=self.timeout)
            proc.stdin.write(json.dumps(job))
            proc.stdin.write_eof()
            proc.stdin.drain()
            out = proc.stdout.read()
            err = proc.stderr.read()
            code_ = proc.wait()
        finally:
            sb.terminate()
        return finish(protocol.parse_stdout(out), key, nonce, code_, err or "",
                      round(time.time() - t0, 2), self.name, seed)

    def run_parallel(self, code, scenario, seeds=(0, 1, 2), enforce_gate=True, scene_graph=None):
        with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
            futs = [ex.submit(self.run, code, scenario, s, enforce_gate, scene_graph) for s in seeds]
            return [f.result() for f in futs]


if __name__ == "__main__":
    # quick smoke test: python -m aegis.sandbox.modal_backend
    from aegis.models import policies

    sb = ModalSandbox()
    res = sb.run(policies.render(policies.SAFE_DEFAULTS), "collision", 0)
    print(res["status"], res["verify_msg"], res["elapsed_s"], "s")
