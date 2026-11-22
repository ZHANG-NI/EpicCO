#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""token_count.py <camp dir> [<camp dir> ...]: token use of the language-model sessions of one or more agent lines,
read from the session transcripts the command-line client keeps (<camp>/<role>/.claude/projects/**/*.jsonl; roles =
the sub-directories of the camp that have such a folder, e.g. enh, gen).

Every assistant message of a transcript carries a "usage" record (input tokens, output tokens, tokens written to and
read from the prompt cache, ...). The client writes the same message several times while it streams, so messages are
de-duplicated by their id and the LARGEST value of each field is kept. All numeric fields of "usage" are summed as
they are (nested ones as a.b), so nothing is dropped; no price is applied here.

Writes tokens/<camp name>.json under the current directory (per role, per transcript file: model, messages, first and
last time stamp, the sums) and prints one line per camp and role. File work only (no solver, no build).
   python3 cluster/token_count.py camp/lmr2_G1 camp/lmr2_S1"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def flat(u, pre=""):
    out = {}
    for k, v in (u or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[pre + k] = v
        elif isinstance(v, dict):
            out.update(flat(v, pre + k + "."))
    return out


def one_file(p):
    msgs, first, last, bad = {}, None, None, 0
    with open(p, encoding="utf8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            ts = r.get("timestamp")
            if ts:
                first = ts if first is None or ts < first else first
                last = ts if last is None or ts > last else last
            m = r.get("message")
            if r.get("type") != "assistant" or not isinstance(m, dict) or not isinstance(m.get("usage"), dict):
                continue
            mid = m.get("id") or r.get("uuid")
            u = flat(m["usage"])
            cur = msgs.setdefault(mid, {"model": m.get("model"), "usage": {}})
            for k, v in u.items():
                if v > cur["usage"].get(k, -1):
                    cur["usage"][k] = v
    tot, models = defaultdict(float), defaultdict(int)
    for x in msgs.values():
        models[x["model"] or "?"] += 1
        for k, v in x["usage"].items():
            tot[k] += v
    return {"file": p.name, "messages": len(msgs), "models": dict(models), "first": first, "last": last,
            "unreadable_lines": bad, "usage": {k: int(v) if float(v).is_integer() else v for k, v in sorted(tot.items())}}


def main():
    outdir = Path("tokens")
    outdir.mkdir(exist_ok=True)
    KEYS = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
    print("%-12s %-4s %8s %9s %13s %13s %15s %15s  %s" % ("camp", "role", "sessions", "messages", "input", "output",
                                                         "cache written", "cache read", "models"))
    for a in sys.argv[1:]:
        camp = Path(a)
        res = {"camp": camp.name, "roles": {}}
        for role in sorted(d for d in camp.iterdir() if d.is_dir() and (d / ".claude" / "projects").is_dir()):
            files = sorted((role / ".claude" / "projects").rglob("*.jsonl"))
            rows = [one_file(p) for p in files]
            tot, models = defaultdict(float), defaultdict(int)
            for r in rows:
                for k, v in r["usage"].items():
                    tot[k] += v
                for k, v in r["models"].items():
                    models[k] += v
            res["roles"][role.name] = {"sessions": len(rows), "messages": sum(r["messages"] for r in rows),
                                       "models": dict(models), "usage": {k: int(v) for k, v in sorted(tot.items())},
                                       "first": min([r["first"] for r in rows if r["first"]] or [None]),
                                       "last": max([r["last"] for r in rows if r["last"]] or [None]), "files": rows}
            print("%-12s %-4s %8d %9d %13d %13d %15d %15d  %s" % (
                camp.name, role.name, len(rows), sum(r["messages"] for r in rows), *[int(tot.get(k, 0)) for k in KEYS],
                ",".join("%s:%d" % kv for kv in sorted(models.items()))))
        with open(outdir / (camp.name + ".json"), "w", encoding="utf8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
