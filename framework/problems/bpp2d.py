"""Two-dimensional bin packing with guillotine cuts (identical or variable-sized bins): instance parsing, independent
solution checking, build and run commands.

Instance file (JSON): {"Name", "Objects": [{"Length", "Height", "Stock" (null = unlimited), "Cost"}],
"Items": [{"Length", "Height", "Demand"}]}. Every item must be cut from a bin; the cuts go from edge to edge
(guillotine cuts). Objective here: the total AREA of the bins used (the "Cost" field is not used).

Solution file (JSON, written by the solver): the instance plus "CuttingPatterns": one entry per bin used,
{"Object": index of the bin type, "Root": node}; a node is {"Length", "Height", "Type": "Structure" | "Item" |
"Leftover", "Item": index of the item type (Item nodes), "Children": [nodes]}. The children of a Structure node
partition it by parallel guillotine cuts: either all have the node's Height and their Lengths add up to its Length,
or all have its Length and their Heights add up to its Height.

The check is independent of the solver: it walks every tree, rejects anything that is not such a partition, a root
that is not the bin, an Item node whose size is not the item's (turned by 90 degrees only when rotation is allowed),
a demand not met exactly, more bins of a type than its stock, and recomputes the area."""
import hashlib
import json
import os
import re

NAME = "bpp2d"
OBJECTIVE = "min"
INSTANCE_GLOB = "*.json"
TIME_LIMIT_S = 60.0

# Files of the host that start its worker thread, enforce the time limit and write the solution: they must stay
# byte-identical to the original (md5); the judge's one-core scan skips them and refuses a tree in which they differ.
LOCKED = {
    "src/gdrr_main.rs": "0b0cc2e412a300768010733f4b8e3c24",
    "src/optimization/sol_collectors/global_sol_collector.rs": "6696f56c8dd79cc058b2dd9b37544224",
}


def parse(path):
    d = json.load(open(path, encoding="utf-8"))
    objs = [(int(o["Length"]), int(o["Height"]), o.get("Stock")) for o in d["Objects"]]
    items = [(int(i["Length"]), int(i["Height"]), int(i.get("Demand") or 1)) for i in d["Items"]]
    if not objs or not items:
        raise ValueError("no bin type or no item")
    return objs, items


def size(path):
    """Number of items (demands added up): the size used for size bands and reports."""
    return sum(q for _, _, q in parse(path)[1])


def time_limit(n):
    """The same limit for every instance, whatever its size."""
    return TIME_LIMIT_S


def item_area(items):
    return sum(l * h * q for l, h, q in items)


def lower_bound(objs, items):
    """Trivial bound: the bins used must cover the item area."""
    return item_area(items)


def _walk(node, items, rot, counts, depth=0):
    """Returns an error text or None. Iterative on the children, recursive on depth (the trees are shallow)."""
    if depth > 200:
        return "tree deeper than 200 levels"
    L, H, typ, ch = node.get("Length"), node.get("Height"), node.get("Type"), node.get("Children") or []
    if not isinstance(L, int) or not isinstance(H, int) or L < 0 or H < 0:
        return "a node without integer Length / Height"
    if typ == "Item":
        if ch:
            return "an Item node with children"
        k = node.get("Item")
        if not isinstance(k, int) or k < 0 or k >= len(items):
            return "an Item node with a wrong item index"
        l, h, _ = items[k]
        if (L, H) != (l, h) and not (rot and (L, H) == (h, l)):
            return "item %d placed as %d x %d, its size is %d x %d%s" % (k, L, H, l, h, "" if rot else " (rotation is not allowed)")
        counts[k] = counts.get(k, 0) + 1
        return None
    if typ == "Leftover":
        return "a Leftover node with children" if ch else None
    if typ != "Structure":
        return "unknown node type %r" % (typ,)
    if not ch:
        return None                                   # an empty region
    side = all(c.get("Height") == H for c in ch) and sum(c.get("Length", -1) for c in ch) == L
    stack = all(c.get("Length") == L for c in ch) and sum(c.get("Height", -1) for c in ch) == H
    if not (side or stack):
        return "the children of a %d x %d node do not partition it by parallel edge-to-edge cuts" % (L, H)
    for c in ch:
        e = _walk(c, items, rot, counts, depth + 1)
        if e:
            return e
    return None


def verify(inst_path, sol_path, opts=None):
    rot = bool((opts or {}).get("rot"))
    try:
        objs, items = parse(inst_path)
    except Exception as ex:
        return {"ok": False, "cost": None, "msg": "instance: %r" % (ex,)}
    if not os.path.exists(sol_path):
        return {"ok": False, "cost": None, "msg": "no solution file"}
    try:
        d = json.load(open(sol_path, encoding="utf-8"))
        cps = d["CuttingPatterns"]
    except Exception as ex:
        return {"ok": False, "cost": None, "msg": "bad solution file: %r" % (ex,)}
    counts, used, area = {}, {}, 0
    for cp in cps:
        k, root = cp.get("Object"), cp.get("Root")
        if not isinstance(k, int) or k < 0 or k >= len(objs) or not isinstance(root, dict):
            return {"ok": False, "cost": None, "msg": "a cutting pattern with a wrong bin index or without a root"}
        L, H, _ = objs[k]
        if (root.get("Length"), root.get("Height")) != (L, H):
            return {"ok": False, "cost": None, "msg": "a root of %s x %s in a bin of %d x %d" % (root.get("Length"), root.get("Height"), L, H)}
        e = _walk(root, items, rot, counts)
        if e:
            return {"ok": False, "cost": None, "msg": e}
        used[k] = used.get(k, 0) + 1
        area += L * H
    for k, (l, h, q) in enumerate(items):
        if counts.get(k, 0) != q:
            return {"ok": False, "cost": None, "msg": "item %d: %d placed, demand %d" % (k, counts.get(k, 0), q)}
    for k, n in used.items():
        st = objs[k][2]
        if st is not None and n > int(st):
            return {"ok": False, "cost": None, "msg": "bin type %d: %d used, stock %d" % (k, n, int(st))}
    if area < lower_bound(objs, items):
        return {"ok": False, "cost": None, "msg": "bin area below the item area (checker error)"}
    claimed = (d.get("Statistics") or {}).get("MaterialCost")
    return {"ok": True, "cost": area, "msg": "" if claimed is None or int(claimed) == area else "solver reported %s" % claimed}


def validate_instance(path, lo, hi):
    try:
        objs, items = parse(path)
    except Exception as ex:
        return False, "parse: %r" % (ex,)
    n = sum(q for _, _, q in items)
    if n < lo or n > hi:
        return False, "n=%d outside [%d, %d]" % (n, lo, hi)
    if min(min(l, h) for l, h, _ in items) < 1 or min(min(l, h) for l, h, _ in objs) < 1:
        return False, "a side below 1"
    for l, h, _ in items:
        if not any(l <= L and h <= H for L, H, _ in objs):
            return False, "an item that fits in no bin type"
    return True, ""


def features(path):
    objs, items = parse(path)
    n = sum(q for _, _, q in items)
    big = max(L * H for L, H, _ in objs)
    return {"n": n, "item_types": len(items), "bin_types": len(objs),
            "mean_item_share": round(item_area(items) / float(n) / big, 4),
            "limited_stock": int(any(s is not None for _, _, s in objs))}


def canonical(path_in, path_out, rng=None):
    """Re-write an instance in one fixed layout (no name); with `rng` the items are also put in a random order."""
    d = json.load(open(path_in, encoding="utf-8"))
    items = [{"Length": int(i["Length"]), "Height": int(i["Height"]), "Demand": int(i.get("Demand") or 1), "DemandMax": None,
              "Value": int(i["Length"]) * int(i["Height"])} for i in d["Items"]]
    if rng is not None:
        rng.shuffle(items)
    objs = [{"Length": int(o["Length"]), "Height": int(o["Height"]), "Stock": o.get("Stock"), "Cost": int(o["Length"]) * int(o["Height"])}
            for o in d["Objects"]]
    json.dump({"Name": "instance", "Objects": objs, "Items": items}, open(path_out, "w"), indent=0)


def signature(path):
    objs, items = parse(path)
    return (tuple(sorted(objs, key=str)), tuple(sorted(items)))


def sig_similarity(a, b):
    return 1.0 if a == b else 0.0


RANGE_FEATURES = ["mean_item_share"]

# Official build (run inside the source tree, $1 = output directory). `cargo` is the first one on PATH; the vendored
# crates sit next to the toolchain (<toolchain>/vendor_gdrr), so the build needs no network. The launcher `solver`
# gives the host the framework's command line.
BUILD = r'''set -e
set -o pipefail
CARGO=$(command -v cargo) || { echo "cargo not found on PATH"; exit 1; }
TC=$(dirname "$(dirname "$(readlink -f "$CARGO")")")
VEND="$TC/vendor_gdrr"
[ -d "$VEND" ] || { echo "vendored crates not found at $VEND"; exit 1; }
[ -f params.json ] || { echo "params.json is missing"; exit 1; }
python3 -c "import json,sys; json.load(open('params.json'))" || { echo "params.json is not valid JSON"; exit 1; }
if [ -n "$GDRR_TARGET_DIR" ]; then          # a kept build directory (your own builds): the crates are compiled once
  TD="$GDRR_TARGET_DIR"; mkdir -p "$TD"
else                                        # official builds: always from an empty directory
  TD=$(mktemp -d "${TMPDIR:-/tmp}/gdrr_build.XXXXXX")
  trap 'rm -rf "$TD"' EXIT
fi
if [ "$2" = check ]; then                   # compile errors only, no binary
  CARGO_HOME="$TD/home" CARGO_TARGET_DIR="$TD/target" cargo check --release --offline --bin gdrr_main \
    --config 'source.crates-io.replace-with="vendored-sources"' \
    --config "source.vendored-sources.directory=\"$VEND\"" 2>&1 | { grep -v "^ *Compiling \|^ *Checking " || true; } | tail -n 150
  exit 0
fi
CARGO_HOME="$TD/home" CARGO_TARGET_DIR="$TD/target" cargo build --release --offline --bin gdrr_main \
  --config 'source.crates-io.replace-with="vendored-sources"' \
  --config "source.vendored-sources.directory=\"$VEND\"" 2>&1 | { grep -v "^ *Compiling \|^ *Finished " || true; } | tail -n 150
cp "$TD/target/release/gdrr_main" "$1/gdrr_main"
cp params.json "$1/params.json"
cat > "$1/solver" <<'LAUNCHER'
#!/usr/bin/env python3
"""solver <instance> <solution> -t <seconds> -seed <int> [-rot 0|1]: writes the run configuration of the host (time
limit, seed, ONE thread, rotation off unless -rot 1, bins valued by their area, plus the algorithm parameters of
params.json) next to the solution file and runs the binary on it."""
import json
import os
import sys
a = sys.argv[1:]
if len(a) < 2:
    sys.exit("usage: solver <instance> <solution> -t <seconds> -seed <int> [-rot 0|1]")
inst, sol, T, seed, rot = a[0], a[1], 60.0, 0, 0
i = 2
while i + 1 < len(a):
    if a[i] == "-t":
        T = float(a[i + 1])
    elif a[i] == "-seed":
        seed = int(a[i + 1])
    elif a[i] == "-rot":
        rot = int(a[i + 1])
    i += 2
here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "params.json")))
cfg.update(maxRunTime=max(1, int(T)), maxRRIterations=None, nThreads=1, randomSeed=seed, rotationAllowed=bool(rot),
           sheetValuationMode="area")
cfgp = sol + ".config.json"
with open(cfgp, "w") as f:
    json.dump(cfg, f)
os.environ["NO_COLOR"] = "1"
sys.stdout.flush()
os.execv(os.path.join(here, "gdrr_main"), ["gdrr_main", inst, cfgp, sol])
LAUNCHER
chmod +x "$1/solver"
'''


def run_cmd(bindir, inst_path, sol_path, T, seed, opts=None):
    cmd = [os.path.join(bindir, "solver"), inst_path, sol_path, "-t", "%d" % max(1, int(T)), "-seed", str(seed)]
    if (opts or {}).get("rot"):
        cmd += ["-rot", "1"]
    return cmd


_STAMP = re.compile(r"\[(\d+):(\d\d):(\d\d)\].*<complete>.*mat:\s*(\d+)")


def trace(sol_path, stdout_path):
    """(seconds, bin area) for every improvement of the complete solution the host printed (one-second stamps)."""
    out = []
    if os.path.exists(stdout_path):
        for line in open(stdout_path, encoding="utf-8", errors="ignore"):
            m = _STAMP.search(line)
            if m:
                out.append((float(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))), float(m.group(4))))
    return out


def locked_violation(tree):
    """Text when a locked file of `tree` is missing or differs from the original, else None."""
    for rel, md5 in LOCKED.items():
        p = os.path.join(str(tree), rel)
        if not os.path.exists(p):
            return "BUILD REFUSED: %s is missing; it must stay exactly as in orig/." % rel
        if hashlib.md5(open(p, "rb").read()).hexdigest() != md5:
            return ("BUILD REFUSED: %s differs from the original. This file starts the solver's single worker thread, enforces "
                    "the time limit and writes the solution; it must stay exactly as in orig/ (restore it with `cp orig/%s src/%s`)."
                    % (rel, rel, rel))
    return None


if __name__ == "__main__":
    import subprocess
    import sys
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "verify":
        print(json.dumps(verify(a[1], a[2], {"rot": 1} if "--rot" in a else None)))
    elif len(a) >= 2 and a[0] == "validate":
        print(validate_instance(a[1], int(a[2]) if len(a) > 2 else 1, int(a[3]) if len(a) > 3 else 10**6))
    elif len(a) >= 2 and a[0] == "features":
        print(json.dumps(features(a[1])))
    elif (len(a) >= 3 and a[0] == "build") or (len(a) >= 2 and a[0] == "check"):
        # your own build / compile check of a source tree, with the official commands and a kept build directory
        bad = locked_violation(a[1])
        if bad:
            sys.exit(bad)
        env = dict(os.environ)
        env.setdefault("GDRR_TARGET_DIR", os.path.join(os.path.expanduser("~"), "tmp", "gdrr_target"))
        out = os.path.abspath(a[2]) if a[0] == "build" else "/nonexistent"
        if a[0] == "build":
            os.makedirs(out, exist_ok=True)
        sys.exit(subprocess.run(["bash", "-c", BUILD, "build", out] + (["check"] if a[0] == "check" else []), cwd=a[1], env=env).returncode)
    else:
        print("usage: problem.py verify <instance> <solution> [--rot] | validate <instance> [min_n max_n] | features <instance> "
              "| build <source dir> <output dir> | check <source dir>")
