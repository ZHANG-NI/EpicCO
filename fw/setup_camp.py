#!/usr/bin/env python3
"""setup_camp.py <camp_name> <task.json> [--no-fresh] [key=value ...]: create a campaign directory.

Benchmark-exposed campaign (old protocol): the enhancer trains on the public benchmark in task["train_dir"].
Blind campaign (key train_twin=<twin dir>): the enhancer NEVER sees the public benchmark. Its training set is drawn
from an instance source (a certified benchmark twin, or a generator written without the benchmark), one instance per
size of the public benchmark; reference values are the best costs found so far (dynamic).
  rotate={"every_s":1500,"k":25}   replace the k oldest training instances periodically (omit for a fixed set)
  train_copy_from=<other camp>     start from the same initial training set as another campaign
"""
import json
import os
import random
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
    name, task = sys.argv[1], json.loads(Path(sys.argv[2]).read_text().replace("/opt/epicco", jail.SE))
    fresh = "--no-fresh" not in sys.argv
    for kv in sys.argv[3:]:
        if "=" in kv and not kv.startswith("--"):
            k, v = kv.split("=", 1)
            try:
                task[k] = json.loads(v)
            except ValueError:
                task[k] = v
    P = importlib.import_module("problems." + task["problem"])
    cd = SE / "camp" / name
    if cd.exists():
        sys.exit("campaign %s already exists" % name)
    secret = secrets.token_hex(8)
    up = Path(task["upstream"])
    v0 = cd / "priv" / "versions" / "v0" / "src"
    v0.mkdir(parents=True)
    for sub in task.get("src_subdirs", []):
        shutil.copytree(up / sub, v0 / sub)
    for f in task.get("src_files", []):
        if (up / f).exists():
            shutil.copy(up / f, v0 / f)
    blind = bool(task.get("train_twin"))
    if blind:
        import twin as twinmod
        fresh = False
        tdir = cd / "priv" / "train" / "inst"
        tdir.mkdir(parents=True)
        ext = P.INSTANCE_GLOB.replace("*", "")
        if task.get("train_copy_from"):
            src = SE / "camp" / task["train_copy_from"] / "priv" / "train"
            names = json.loads((src / "initial.json").read_text())
            for nm in names:
                shutil.copy(src / "inst" / (nm + ext), tdir / (nm + ext))
                for se in getattr(P, "SIDE_EXT", ()):       # files the plugin's verifier keeps next to an instance (links: they are large)
                    sf = src / "inst" / (nm + ext + se)
                    if sf.exists():
                        os.symlink(os.path.realpath(sf), tdir / (nm + ext + se))
        else:
            T = twinmod.Twin(task["train_twin"])
            sizes = [n for _, _, n in T.train]                      # one training instance per size of the public benchmark
            if task.get("train_sizes"):                             # a small training set of the given sizes
                sizes = [int(x) for x in task["train_sizes"]]
            res = T.draw(len(sizes), tdir, "T00", sizes, random.Random(secret + "-train"))
            names = sorted(nm for nm, d in res.items() if d["ok"])
            print("initial training set: %d of %d drawn instances valid (source generator version %d)" % (len(names), len(sizes), T.st["accepted"]))
        if task.get("train_sizes") and not task.get("train_copy_from"):
            if len(names) != len(task["train_sizes"]):
                sys.exit("only %d of %d training instances could be drawn" % (len(names), len(task["train_sizes"])))
        elif not task.get("train_sizes"):
            random.Random(secret + "-order").shuffle(names)
        (cd / "priv" / "train" / "initial.json").write_text(json.dumps(names))
        (cd / "priv" / "train" / "active.json").write_text(json.dumps(names))
        ref = cd / "priv" / "train_ref.json"
        ref.write_text(json.dumps({nm: None for nm in names}))
        task.update(train_dir=str(tdir), bks=str(ref), dynamic_ref=True)
        task.setdefault("prompt_template", "enhance_blind.md")
    bks = json.loads(Path(task["bks"]).read_text())
    train = sorted(p for p in Path(task["train_dir"]).glob(P.INSTANCE_GLOB) if p.stem in bks)
    ns = [P.size(str(p)) for p in train]
    for who in ("enh", "gen"):
        jail.prepare_home(cd / who)
    W, G = cd / "enh" / "work", cd / "gen" / "work"
    for d in ("src", "orig", "committed"):
        shutil.copytree(v0, W / d)
    for p in (W / "orig").rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    (W / "instances" / "train").mkdir(parents=True)
    (W / "instances" / "fresh").mkdir()
    (G / "public").mkdir(parents=True)
    for p in train:
        shutil.copy(p, W / "instances" / "train" / p.name)
        if not blind:
            shutil.copy(p, G / "public" / p.name)
    (W / "targets.json").write_text(json.dumps({p.stem: bks[p.stem] for p in train}, indent=0))
    for X in (W, G):
        (X / "tools").mkdir()
        shutil.copy(FW / "ctl.py", X / "tools" / "ctl.py")
        shutil.copy(FW / "problems" / (task["problem"] + ".py"), X / "tools" / "problem.py")
        (X / "box" / "req").mkdir(parents=True)
        (X / "box" / "res").mkdir()
        (X / "NOTES.md").write_text("(no notes yet)\n")
    (W / "runs").mkdir()
    for d in ("gen", "scratch", "feedback", "out"):
        (G / d).mkdir()
    C = {k: v for k, v in task.items() if k not in ("upstream", "src_subdirs", "src_files")}
    C.update(name=name, fresh=fresh, pool=os.environ.get("SE_POOL", str(SE / "pool")), n_train=len(train), secret=secret)
    C["prompt"] = {k: v.replace("{lo}", str(min(ns))).replace("{hi}", str(max(ns))) for k, v in task["prompt"].items()}
    (cd / "camp.json").write_text(json.dumps(C, indent=1))
    (cd / "logs").mkdir()
    print("campaign %s: %d training instances, n in [%d, %d], fresh=%s, blind=%s, rotate=%s" % (
        name, len(train), min(ns), max(ns), fresh, blind, task.get("rotate")))


if __name__ == "__main__":
    main()
