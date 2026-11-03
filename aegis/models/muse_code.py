import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from aegis.config import settings
from aegis.models import policies
from aegis.models.meta_client import MetaClient, extract_python
from aegis.sandbox import gate

log = logging.getLogger("aegis.muse_code")

SYSTEM = """You write Python control policies for a simulated Franka Panda arm.
Output one ```python block and nothing else. The module must define exactly:

    def execute_policy(env) -> None

The env object (SafePandaEnv) is the only way to act. Its API:
  env.payload, env.tray            object names (str)
  env.obstacles                    list of obstacle names
  env.scene                        perception scene graph (dict, read only)
  env.get_position(name)           (x, y, z) of the object center, meters
  env.get_size(name)               {"radius", "height"} for cylinders, {"half_extent", "height"} for the tray
  env.ee_position()                fingertip center (x, y, z)
  env.move_to((x, y, z), speed=0.25, yaw=0.0)   straight Cartesian move, top down gripper, speed in m/s (max 0.6)
  env.follow(list_of_xyz, speed=0.25)           chain of move_to calls
  env.open_gripper()
  env.close_gripper(force=20.0)    clamp force per finger in newtons (max 70)
  env.wait(seconds)
  env.log(msg)

Geometry facts: the table is z=0. Grasp a cylinder with the fingertips about 0.03 m below its top.
While carried, the payload bottom hangs about (grasp height) below the fingertips.

Hard rules, enforced by a static gate and a runtime audit hook. Breaking one kills the rollout:
  imports: only math, numpy, typing
  no attribute or name starting with an underscore
  no getattr, setattr, eval, exec, compile, open, globals, vars, exit
  no direct simulator access of any kind. The grader only accepts physically executed motion.
Keep it short, deterministic and readable. Name the phases with env.log("pre_grasp") etc."""


def build_argv(template, task, workdir):
    # split the template first, then fill placeholders, so quoting never mangles the task or the path
    tokens = shlex.split(template, posix=os.name != "nt")
    argv = []
    for tok in tokens:
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            tok = tok[1:-1]
        argv.append(tok.replace("{task}", task).replace("{dir}", workdir))
    exe = shutil.which(argv[0]) if argv else None
    if exe:
        argv[0] = exe  # Windows needs the full path to muse.cmd, plain CreateProcess only finds .exe
    return argv


class MuseCode:
    def __init__(self):
        self.client = None
        self.last_error = None
        self.last_source = None  # who actually produced the last policy
        if settings.code_cmd:
            self.mode = "cli"
        elif settings.backend("code") == "live":
            try:
                self.client = MetaClient()
                self.mode = "api"
            except Exception as e:
                self.last_error = str(e)
                self.mode = "offline"
        else:
            self.mode = "offline"

    @property
    def name(self):
        return {"cli": "Muse Code (CLI)", "api": f"Muse Code via Model API ({settings.code_model})",
                "offline": "offline policy writer"}[self.mode]

    def _agent(self, user_msg, current_code):
        # Muse Code works like an engineer: it gets a scratch repo with policy.py and TASK.md and edits the file
        with tempfile.TemporaryDirectory(prefix="aegis_muse_") as d:
            pol = Path(d) / "policy.py"
            original = current_code or "def execute_policy(env):\n    pass\n"
            pol.write_text(original, encoding="utf-8")
            (Path(d) / "TASK.md").write_text(SYSTEM + "\n\n" + user_msg, encoding="utf-8")
            task = "Read TASK.md and rewrite policy.py so it satisfies it. Edit only policy.py."
            argv = build_argv(settings.code_cmd, task, d)
            proc = subprocess.run(argv, cwd=d, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=900)
            edited = pol.read_text(encoding="utf-8")
            if edited != original and "def execute_policy" in edited:
                return edited
            if "def execute_policy" in (proc.stdout or ""):
                return extract_python(proc.stdout)
            raise RuntimeError(f"Muse Code exit {proc.returncode}, policy.py unchanged: {(proc.stderr or proc.stdout)[-300:]}")

    def _ask(self, user_msg, current_code=""):
        if self.mode == "cli":
            return self._agent(user_msg, current_code)
        txt = self.client.chat(settings.code_model, [{"role": "developer", "content": SYSTEM},
                                                     {"role": "user", "content": user_msg}],
                               max_tokens=6000)
        return extract_python(txt)

    def synthesize(self, instruction, plan, scene_graph, scenario_key, params=None):
        # returns (code, params). params are only meaningful for the offline writer.
        if self.mode != "offline":
            msg = json.dumps({"instruction": instruction, "plan": plan, "scene_graph": scene_graph}, indent=1)
            try:
                code = self._ask("Write the policy.\n" + msg)
                self.last_source = self.name
                return code, None
            except Exception as e:
                self.last_error = str(e)
                log.warning("Muse Code synthesis failed (%s), using offline writer", e)
        params = params or dict(policies.SAFE_DEFAULTS)
        self.last_source = self._fallback_label()
        return policies.render(params), params

    def patch(self, code, diagnosis, params, scene_graph):
        if self.mode != "offline":
            msg = ("This policy failed verification. Fix the root cause and return the full corrected module.\n"
                   f"Diagnosis:\n{json.dumps(diagnosis, indent=1)}\n\nCurrent code:\n```python\n{code}```")
            for _ in range(2):
                try:
                    new = self._ask(msg, code)
                    report = gate.check(new)
                    if report.passed:
                        self.last_source = self.name
                        return new, None
                    msg += f"\n\nYour last answer was rejected by the AST gate: {report.summary()}"
                except Exception as e:
                    self.last_error = str(e)
                    log.warning("Muse Code patch failed: %s", e)
                    break
        base = params if params else dict(policies.SAFE_DEFAULTS)
        new_params = policies.apply_directives(base, diagnosis.get("patch_directives", []))
        self.last_source = self._fallback_label()
        return policies.render(new_params), new_params

    def _fallback_label(self):
        if self.mode == "offline":
            return "offline policy writer"
        return f"offline policy writer ({self.name} failed: {(self.last_error or 'no usable code')[:120]})"
