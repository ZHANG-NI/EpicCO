#!/usr/bin/env python3
"""setup_twin.py <twin_name> <task.json> [key=value ...]: create a benchmark-twin directory (phase 0) from a task."""
import json
import secrets
import shutil
import sys
from pathlib import Path

FW = Path(__file__).resolve().parent
sys.path.insert(0, str(FW))
import importlib   # noqa: E402
import jail        # noqa: E402

SE = Path(jail.SE)


def main():
    name, task = sys.argv[1], json.loads(Path(sys.argv[2]).read_text().replace("/opt/alphaco", jail.SE))
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        try:
            task[k] = json.loads(v)
        except ValueError:
            task[k] = v
    P = importlib.import_module("problems." + task["problem"])
    td = SE / "twin" / name
    if td.exists():
        sys.exit("twin %s already exists" % name)
    up = Path(task["upstream"])
    src = td / "priv" / "solver_src"
    src.mkdir(parents=True)
    for sub in task.get("src_subdirs", []):
        shutil.copytree(up / sub, src / sub)
    for f in task.get("src_files", []):
        if (up / f).exists():
            shutil.copy(up / f, src / f)
    train = sorted(Path(task["train_dir"]).glob(P.INSTANCE_GLOB))
    ns = [P.size(str(p)) for p in train]
    for who in ("gen", "dis", "genrun", "disrun"):
        jail.prepare_home(td / who)
    G, D = td / "gen" / "work", td / "dis" / "work"
    G.mkdir(parents=True, exist_ok=True)
    if not task.get("naive"):                 # a naive generator is written WITHOUT the public files
        (G / "public").mkdir(parents=True)
        for p in train:
            shutil.copy(p, G / "public" / p.name)
    for X in (G, D):
        (X / "tools").mkdir(exist_ok=True)
        shutil.copy(FW / "ctl.py", X / "tools" / "ctl.py")
        (X / "box" / "req").mkdir(parents=True)
        (X / "box" / "res").mkdir()
        (X / "NOTES.md").write_text("(no notes yet)\n")
    shutil.copy(FW / "problems" / (task["problem"] + ".py"), G / "tools" / "problem.py")
    for d in ("gen", "scratch", "feedback"):
        (G / d).mkdir()
    (D / "scratch").mkdir()
    C = {"name": name, "problem": task["problem"], "train_dir": task["train_dir"], "train_opts": task.get("train_opts", {}),
         "solver_tree": str(src), "pool": str(SE / "pool"), "secret": secrets.token_hex(8), "n_train": len(train),
         "model": task.get("model", "claude-opus-5-5"), "acc_threshold": task.get("acc_threshold", 0.60),
         "max_submits": task.get("max_submits", 3), "max_rounds": task.get("max_rounds", 5),
         "time_scale": task.get("time_scale", 1.0), "naive": bool(task.get("naive", False)), "lo": min(ns), "hi": max(ns), "session_timeout_s": task.get("twin_session_timeout_s", 2700)}
    C["prompt"] = {k: v.replace("{lo}", str(min(ns))).replace("{hi}", str(max(ns))) for k, v in task["prompt"].items()}
    (td / "twin.json").write_text(json.dumps(C, indent=1))
    (td / "logs").mkdir()
    (td / "cmd").mkdir()
    print("twin %s: %d public instances, n in [%d, %d]" % (name, len(train), min(ns), max(ns)))


if __name__ == "__main__":
    main()
