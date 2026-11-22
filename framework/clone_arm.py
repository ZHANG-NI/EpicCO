#!/usr/bin/env python3
"""clone_arm.py <source_camp> <new_camp> <task.json> [setup_camp key=value ...]

Create a new blind campaign that starts from exactly the same initial training set as <source_camp> and re-uses the
source's calibration of the original solver on those instances (same solver source, same instances, same seeds), so
that no core time is spent on measuring it again. Reference values are reset to the calibration state."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

FW = Path(__file__).resolve().parent
sys.path.insert(0, str(FW))
import jail                      # noqa: E402
from judge import tree_hash      # noqa: E402

SE = Path(jail.SE)


def main():
    src, new, task = sys.argv[1], sys.argv[2], sys.argv[3]
    S, N = SE / "camp" / src, SE / "camp" / new
    sc = json.loads((S / "camp.json").read_text())
    r = subprocess.run([sys.executable, str(FW / "setup_camp.py"), new, task, "train_twin=" + sc["train_twin"],
                        "train_copy_from=" + src] + sys.argv[4:], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        sys.exit(r.stderr[-2000:])
    names = json.loads((N / "priv" / "train" / "initial.json").read_text())
    calib = [json.loads(l) for l in open(S / "priv" / "ledger.jsonl") if '"type": "calib"' in l][-1]
    h = calib["hash"]
    assert tree_hash(N / "priv" / "versions" / "v0" / "src") == h, "the original solver source differs from the source campaign's"
    cache = json.loads((S / "priv" / "cache" / (h + ".json")).read_text())
    keep = {k: v for k, v in cache.items() if v["grp"] == "train" and v["name"] in names and k.startswith("train__")}
    (N / "priv" / "cache").mkdir(parents=True, exist_ok=True)
    (N / "priv" / "cache" / (h + ".json")).write_text(json.dumps(keep))
    ref = {nm: None for nm in names}
    for v in keep.values():
        if v["ok"] and (ref[v["name"]] is None or v["cost"] < ref[v["name"]]):
            ref[v["name"]] = v["cost"]
    (N / "priv" / "train_ref.json").write_text(json.dumps(ref))
    (N / "enh" / "work" / "targets.json").write_text(json.dumps(ref, indent=0))
    ss = json.loads((S / "priv" / "state.json").read_text())
    st = {"version": 0, "hash": h, "evals": 0, "batches": [], "active": [], "gen_version": 0, "last_eval": None,
          "eps_train": ss["eps_train"], "eps_fresh": ss["eps_fresh"], "t0": None, "v0_seed_means": ss.get("v0_seed_means"),
          "train_order": names}
    (N / "priv" / "state.json").write_text(json.dumps(st, indent=1))
    with open(N / "priv" / "ledger.jsonl", "a") as f:
        f.write(json.dumps(calib) + "\n")
    shutil.copy(S / "logs" / "calib.out", N / "logs" / "calib.out")
    print("%s: %d training instances, %d cached runs of the original, %d references set, eps %.4f" % (
        new, len(names), len(keep), sum(v is not None for v in ref.values()), st["eps_train"]))


if __name__ == "__main__":
    main()
