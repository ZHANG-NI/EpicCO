#!/usr/bin/env python3
"""set_opts.py <camp_dir> key=value [...]: change options of an existing campaign (camp.json) before it is resumed,
e.g. budget_h=18, session_cap_s=7200, 'rotate={"every_s":3600,"k":25}', 'v4.t_accept=2.0' (dotted key = one field of
a dict option). Values are JSON when they parse as JSON. The driver and the judge read camp.json when they start."""
import json
import os
import sys
import time
from pathlib import Path

cd = Path(sys.argv[1])
p = cd / "camp.json"
C = json.loads(p.read_text())
changes = []
for kv in sys.argv[2:]:
    if "=" not in kv:
        continue
    k, v = kv.split("=", 1)
    try:
        v = json.loads(v)
    except ValueError:
        pass
    d, parts = C, k.split(".")
    for q in parts[:-1]:
        d = d.setdefault(q, {})
    if d.get(parts[-1]) != v:
        changes.append((k, d.get(parts[-1]), v))
        d[parts[-1]] = v
if changes:
    tmp = cd / "camp.json.tmp"
    tmp.write_text(json.dumps(C, indent=1))
    os.replace(tmp, p)
    with open(cd / "priv" / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"type": "options", "t": time.time(), "changes": [[k, a, b] for k, a, b in changes]}) + "\n")
for k, a, b in changes:
    print("option %s: %r -> %r" % (k, a, b))
if not changes:
    print("options unchanged")
