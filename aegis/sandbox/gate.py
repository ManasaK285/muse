import ast
import re
from dataclasses import dataclass, field

ALLOWED_IMPORTS = {"math", "numpy", "typing"}

BANNED_CALLS = {
    "getattr", "setattr", "delattr", "eval", "exec", "compile", "__import__", "globals",
    "locals", "vars", "open", "input", "exit", "quit", "breakpoint", "memoryview", "help",
}
BANNED_NAMES = {
    "SystemExit", "KeyboardInterrupt", "os", "sys", "subprocess", "ctypes", "gc", "inspect",
    "importlib", "builtins", "pybullet", "socket", "signal", "threading", "multiprocessing",
}
TELEPORT_API = {
    "resetBasePositionAndOrientation", "resetJointState", "resetJointStatesMultiDof",
    "resetBaseVelocity", "changeDynamics", "applyExternalForce", "applyExternalTorque",
    "createConstraint", "changeConstraint", "removeBody", "setGravity", "restoreState",
    "setPhysicsEngineParameter", "setTimeStep", "loadURDF", "createMultiBody",
}
SUSPICIOUS_STRING = re.compile(r"reset|teleport|BasePosition|JointState|restoreState|__\w+__", re.I)


@dataclass
class Violation:
    line: int
    rule: str
    detail: str

    def to_dict(self):
        return {"line": self.line, "rule": self.rule, "detail": self.detail}


@dataclass
class GateReport:
    passed: bool
    violations: list = field(default_factory=list)

    def to_dict(self):
        return {"passed": self.passed, "violations": [v.to_dict() for v in self.violations]}

    def summary(self):
        if self.passed:
            return "AST integrity gate passed"
        return "; ".join(f"L{v.line} {v.rule}: {v.detail}" for v in self.violations)


def _fold_str(node):
    # best effort constant folding so "reset" + "BasePosition..." is visible to the gate
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _fold_str(node.left), _fold_str(node.right)
        if a is not None and b is not None:
            return a + b
    if isinstance(node, ast.JoinedStr):
        parts = [v.value for v in node.values if isinstance(v, ast.Constant)]
        return "".join(parts)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join" and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))):
        sep = _fold_str(node.func.value) or ""
        items = [_fold_str(e) for e in node.args[0].elts]
        if all(i is not None for i in items):
            return sep.join(items)
    return None


class _Checker(ast.NodeVisitor):
    def __init__(self):
        self.v = []

    def flag(self, node, rule, detail):
        self.v.append(Violation(getattr(node, "lineno", 0), rule, detail))

    def visit_Import(self, node):
        for a in node.names:
            root = a.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                self.flag(node, "FORBIDDEN_IMPORT", f"import {a.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        root = (node.module or "").split(".")[0]
        if node.level or root not in ALLOWED_IMPORTS:
            self.flag(node, "FORBIDDEN_IMPORT", f"from {node.module} import ...")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("_"):
            self.flag(node, "PRIVATE_ATTRIBUTE", f".{node.attr}")
        if node.attr in TELEPORT_API:
            self.flag(node, "STATE_TELEPORTATION", f".{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in BANNED_CALLS:
            self.flag(node, "DYNAMIC_DISPATCH", node.id)
        elif node.id in BANNED_NAMES:
            self.flag(node, "FORBIDDEN_NAME", node.id)
        elif node.id.startswith("__"):
            self.flag(node, "DUNDER_NAME", node.id)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and SUSPICIOUS_STRING.search(node.value):
            self.flag(node, "OBFUSCATED_API_STRING", repr(node.value[:60]))

    def visit_BinOp(self, node):
        folded = _fold_str(node)
        if folded and any(api in folded for api in TELEPORT_API):
            self.flag(node, "STATE_TELEPORTATION", f"string assembles {folded!r}")
        self.generic_visit(node)

    def visit_Global(self, node):
        self.flag(node, "SCOPE_ESCAPE", "global statement")

    def visit_Nonlocal(self, node):
        self.flag(node, "SCOPE_ESCAPE", "nonlocal statement")

    def visit_Raise(self, node):
        exc = node.exc
        name = getattr(getattr(exc, "func", exc), "id", "")
        if name in ("SystemExit", "KeyboardInterrupt", "GeneratorExit"):
            self.flag(node, "EXIT_SPOOF", f"raise {name}")
        self.generic_visit(node)


def check(code: str) -> GateReport:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return GateReport(False, [Violation(e.lineno or 0, "SYNTAX_ERROR", str(e.msg))])
    c = _Checker()
    c.visit(tree)

    entry = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_policy"]
    if not entry:
        c.v.append(Violation(0, "MISSING_ENTRYPOINT", "def execute_policy(env) not found"))
    elif len(entry[0].args.args) != 1:
        c.v.append(Violation(entry[0].lineno, "BAD_SIGNATURE", "execute_policy must take exactly (env)"))

    # top level may only hold imports, defs and plain constant assignments
    for n in tree.body:
        ok = isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign))
        ok = ok or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
        if not ok:
            c.v.append(Violation(n.lineno, "TOP_LEVEL_SIDE_EFFECT", type(n).__name__))
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            for sub in ast.walk(n.value) if n.value else []:
                if isinstance(sub, ast.Call):
                    c.v.append(Violation(n.lineno, "TOP_LEVEL_SIDE_EFFECT", "call at import time"))
                    break

    seen = set()
    uniq = []
    for v in c.v:
        k = (v.line, v.rule, v.detail)
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    return GateReport(not uniq, uniq)
