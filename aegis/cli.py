import argparse
import json
import logging
import sys
import time

from aegis import scenarios
from aegis.config import settings

C = {"red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m", "dim": "\033[2m",
     "bold": "\033[1m", "off": "\033[0m"}


def paint(txt, color):
    return f"{C[color]}{txt}{C['off']}" if sys.stdout.isatty() else txt


def printer(kind, d):
    ts = time.strftime("%H:%M:%S")
    line = None
    if kind == "mission_start":
        line = paint(f"MISSION {d['title']}  chaos={d['chaos']}", "bold") + "\n  " + d["instruction"]
        line += "\n  " + paint(json.dumps(d["backends"]), "dim")
    elif kind == "node" and d["status"] == "start":
        line = paint(f"-> node {d['node']} (iteration {d['iteration']})", "cyan")
    elif kind == "plan":
        line = "   plan: " + " > ".join(s["name"] for s in d["plan"].get("subgoals", [])) + f"  ({d['planner']})"
    elif kind == "wonder":
        pl = d["prediction"]["objects"]
        payload = next((n for n, o in pl.items() if not o.get("must_not_move")), None)
        line = f"   world model {d['world_model']}: expects {payload} at {pl.get(payload, {}).get('centroid')}"
        line += f", {sum(1 for o in pl.values() if o.get('must_not_move'))} objects fixed"
        line += f"\n   goal frame from {d['source']}"
        if d.get("error"):
            line += paint(f"  (fallback reason: {d['error'][:160]})", "yellow")
    elif kind == "code":
        line = f"   policy v{d['iteration']} from {d['source']}"
        if d.get("directives"):
            line += f"  directives={json.dumps(d['directives'])}"
    elif kind == "gate":
        line = paint("   AST gate: PASS", "green") if d["passed"] else paint(
            "   AST gate: BLOCK " + "; ".join(f"L{v['line']} {v['rule']}" for v in d["violations"]), "red")
    elif kind == "redteam_probe":
        line = paint(f"   red team probe (static gate off): {d['status']} -> "
                     f"{(d.get('violation') or {}).get('message', '')} [{d['verify_msg']}]", "yellow")
    elif kind in ("rollout", "naive"):
        t = d.get("telemetry") or {}
        tag = "NAIVE " if kind == "naive" else f"seed {d['seed']} "
        line = f"   {tag}{d['status']:<18} exit={d['exit_code']} {d['verify_msg']:<22}"
        if t:
            line += (f" contact={t.get('max_contact_force_n')} goal_err={t.get('goal_error_m')}"
                     f" drops={t.get('payload_drops')}")
        if kind == "naive":
            line += "\n   " + paint(f"naive harness verdict: {d['naive_verdict']}", "yellow")
    elif kind == "forecast":
        fc = d["forecast"]
        line = paint(f"   Wonder pre-rollout forecast v{d['iteration']}: {fc.get('predicted_outcome')} "
                     f"({fc.get('confidence')}) {fc.get('reasoning', '')[:200]}", "cyan")
    elif kind == "forecast_score":
        line = paint(f"   forecast check v{d['iteration']}: predicted {d['predicted']}, physics says {d['actual']} "
                     f"-> {'HIT' if d['hit'] else 'MISS'}", "green" if d["hit"] else "yellow")
    elif kind == "audit":
        col = "green" if d["passed"] else "red"
        line = paint(f"   counterfactual audit: survival {d['survival']}", col)
        for f in d["findings"][:6]:
            line += "\n     - " + f["detail"]
        for a in d.get("advisories", [])[:3]:
            line += "\n     ~ " + a
        st = d.get("sam_stats")
        if st:
            col = "green" if st["hit_rate"] >= 0.8 else "yellow"
            line += paint(f"\n     SAM 3.1 so far: {st['objects_found']} found / {st['objects_missed']} missed "
                          f"({st['hit_rate']:.0%}) in {st['calls']} calls", col)
            if st["found_with_alt_phrase"]:
                line += paint(f"\n     phrases that worked: {sorted({k.split(' <- ')[1] for k in st['found_with_alt_phrase']})}", "dim")
            if st["missed_by_object"]:
                line += paint(f"; renderer filled misses for {st['missed_by_object']}", "yellow")
    elif kind == "redteam":
        col = "green" if d["blocked"] == d["total"] else "red"
        line = paint(f"   RED TEAM ({d['source']}): {d['blocked']}/{d['total']} attacks blocked", col)
        if d.get("error"):
            line += paint(f"  (attacker fallback: {d['error'][:150]})", "yellow")
        for a in d["attacks"]:
            who = ", ".join(a["caught_by"]) or "NOTHING (breach)"
            line += f"\n     - {a['name']:<26} caught by: {who}"
    elif kind == "sandbox_error":
        line = paint(f"   HARNESS ERROR on {d['platform']}: {d['cause']}", "red")
        if d.get("stderr_tail"):
            line += "\n" + "\n".join("     | " + ln for ln in d["stderr_tail"].strip().splitlines()[-12:])
    elif kind == "diagnosis":
        dg = d["diagnosis"]
        line = paint(f"   diagnosis {dg['failure_mode']}: {dg['root_cause']}", "yellow")
    elif kind == "verdict":
        col = "green" if d["verdict"] == "VERIFIED" else "red"
        line = paint(f"VERDICT {d['verdict']}  survival={d['survival']}  recovery_cycles={d['recovery_cycles']}  "
                     f"naive said: {d['naive_verdict']}  ({d['elapsed_s']}s)", col)
    if line:
        print(f"[{ts}] {line}", flush=True)


def doctor():
    import platform

    from aegis.graph.mission import Toolkit
    from aegis.models import policies

    print(f"platform    {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"python      {sys.version.split()[0]} at {sys.executable}")
    tk = Toolkit(lambda *_: None)
    for k, v in tk.backends().items():
        print(f"{k:<12}{v}")
    print("\nMeta model stack:")
    for m in tk.model_stack():
        col = "green" if m["state"] == "live" else "yellow"
        state = f"{m['state']:<20}"
        print(f"  {m['model']:<22} " + paint(state, col) + " " + m["role"])
    import shutil
    muse = shutil.which("muse")
    print(f"\nmuse CLI    {muse or 'not on PATH'}" + ("" if settings.code_cmd else "  (MUSE_CODE_CMD not set, see README)"))
    print(f"api key     {'set' if settings.api_key else 'not set'} ({settings.base_url})")
    print("\nsandbox self test: one safe rollout of the collision scene ...", flush=True)
    r = tk.sandbox.run(policies.render(policies.SAFE_DEFAULTS), "collision", 0)
    tel = (r.get("report") or {}).get("telemetry") or {}
    if r["verified"] and r["status"] == "ok":
        print(paint(f"PASS  {r['verify_msg']}, {tel.get('steps')} physics steps, "
                    f"goal error {tel.get('goal_error_m')} m, {r['elapsed_s']} s", "green"))
        return 0
    print(paint(f"FAIL  status={r['status']} exit={r['exit_code']} {r['verify_msg']}", "red"))
    if r.get("stderr_tail"):
        print("sandbox stderr:")
        print(r["stderr_tail"].strip()[-3000:])
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aegis", description="Aegis-VLA verification harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one mission")
    r.add_argument("--scenario", "-s", default="collision", choices=list(scenarios.SCENARIOS) + ["all"])
    r.add_argument("--no-chaos", action="store_true", help="let Muse Code write iteration 1 instead of injecting the flaw")
    r.add_argument("--max-iters", type=int, default=settings.max_iters)
    r.add_argument("--no-media", action="store_true", help="skip GIF and image export")
    r.add_argument("--redteam", action="store_true", help="after the verdict, attack the grader with generated exploits")
    r.add_argument("--voice", metavar="AUDIO", help="speak the instruction: transcribe this audio file with Muse Voice Transcribe")
    r.add_argument("--instruction", help="override the scenario's typed instruction")
    s = sub.add_parser("serve", help="start the mission control dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("doctor", help="show which backends are live")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("aegis.server.app:app", host=args.host, port=args.port, log_level="info")
        return 0
    if args.cmd == "doctor":
        return doctor()

    from aegis.graph.mission import run_mission
    from aegis.recorder import Recorder
    keys = list(scenarios.SCENARIOS) if args.scenario == "all" else [args.scenario]
    instruction = args.instruction
    if args.voice:
        from pathlib import Path

        from aegis.models import voice
        path = Path(args.voice)
        instruction, backend = voice.transcribe(path.read_bytes(), path.name)
        print(paint(f"[voice] {backend}: \"{instruction}\"", "cyan"))
    worst = 0
    for key in keys:
        rec = Recorder(key, save_media=not args.no_media)

        def emit(kind, data):
            printer(kind, data)
            rec(kind, data)

        _, summary = run_mission(key, chaos=not args.no_chaos, emit=emit, max_iters=args.max_iters,
                                 instruction=instruction, redteam=args.redteam)
        rec.close()
        print(f"   artifacts: {rec.dir}\n")
        worst = max(worst, 0 if summary["verdict"] == "VERIFIED" else 1)
        if summary["verdict"] == "HARNESS_ERROR":
            print(paint("   stopping: the sandbox itself is failing, see the stderr above or run "
                        "`python -m aegis doctor`", "red"))
            break
    return worst


if __name__ == "__main__":
    sys.exit(main())
