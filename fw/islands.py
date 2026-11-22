#!/usr/bin/env python3
"""Islands: several campaigns improve the same original solver in parallel, each on its own FIXED training instances.
One island is the GENERAL island (camp.json role "generalist": instances of every size; its committed version is the
deliverable); the others are SCOUTS (role "specialist": part of the size range). Topology: a hub. Nothing is ever
tested twice:

  down, brief       REF_<general>.md in the scout's workspace: what the general island kept, what it tried and did
                    NOT keep, what it is working on now - all three for information only and off limits for the scout
                    (no porting, no re-testing, no duplication). Nothing is copied into the scout's version: a scout
                    keeps its own line (original + its own commits); its role is to give the general island ideas.
  down, version     (camp option sync_main of a scout, 11-13) between two sessions of a scout its committed version
                    is set to the general island's (judge.apply_sync, ledger record "rebase"): the scout then works
                    on the solver as it is now, its offers are diffs against a recent general version, and what it
                    measures holds there. Not while one of its offers is still open (at most sync_max_skips times).
  up, offers        every change a scout commits is put into the general island's workspace as an offer
                    (offers/<scout>_v<N>/change.diff + note.md with what it measured on the scout's instances). The
                    general island's agent applies offers one at a time; an offer becomes part of the deliverable only
                    if it passes the general island's gate (all sizes). Then it flows down to all scouts by itself.
  up, brief         REF_<scout>.md in the general island's workspace: what that scout tried without keeping.

The driver calls refresh_refs(camp) at every session start and every `migrate_poll_s` seconds (cheap: it does something
only when a peer has something new).

  python3 islands.py refs <camp>                 one migration step for this island, now
  python3 islands.py brief <camp>                the brief a scout would get about <camp> (read-only test)
  python3 islands.py carry <camp> <from_camp>    start an island from another campaign's committed version (after the
                                                 calibration, before the driver): original stays v0, carried tree = v1
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jail   # noqa: E402

SE = Path(jail.SE)


def _ledger(cd, rec):
    rec["t"] = time.time()
    with open(cd / "priv" / "ledger.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def _json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def _rows(cd):
    out = []
    try:
        for line in open(cd / "priv" / "ledger.jsonl"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _put(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, p)


def _label(camp):
    return camp.split("_")[-1]


def island_note(camp):
    """One sentence on what an island trains on."""
    C = _json(SE / "camp" / camp / "camp.json", {})
    if C.get("island_note"):
        return C["island_note"]
    ts = C.get("train_sizes")
    return ("its training instances have %d to %d customers" % (min(ts), max(ts))) if ts else "its training instances cover every size"


_clean = lambda t: re.sub(r"\b[A-Z]\d\d-\d+(-n\d+)?\b", "<instance>", " ".join((t or "").split()))[:260]   # noqa: E731


def _group(camp):
    """Name of the island group = the set of islands that talk to each other (this island and its peers)."""
    C = _json(SE / "camp" / camp / "camp.json", {})
    return C.get("group") or "+".join(sorted(set([camp] + list(C.get("peers", [])))))


def _board(camp):
    d = SE / "islands"
    d.mkdir(exist_ok=True)
    return d / (_group(camp) + ".jsonl")


def claims(camp):
    """Latest declared direction of every island of the group: {island: {"text", "t"}}."""
    out = {}
    try:
        for line in open(_board(camp)):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out[r["camp"]] = r
    except OSError:
        pass
    return out


CLAIM_RULE = 2      # version of the declaration protocol; the judge asks for a new declaration when it changes


def claim_history(camp):
    """Every declaration on the group's board, oldest first: {island: [{"text", "t"}, ...]}."""
    out = {}
    try:
        for line in open(_board(camp)):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out.setdefault(r["camp"], []).append(r)
    except OSError:
        pass
    return out


def _others_record(camp):
    """For the answer to a declaration: per other island of the group, what it is working on now, what it declared
    earlier and left, what it kept, what it tried and did not keep (from its ledger). Returns text lines."""
    C = _json(SE / "camp" / camp / "camp.json", {})
    hist, L = claim_history(camp), []
    for other in C.get("peers", []):
        h = hist.get(other, [])
        body = brief(_rows(SE / "camp" / other), downward=True, rule=False)
        if not h and not body:
            continue
        L += ["", "=== project `%s` ===" % _label(other)]
        if h:
            L += ["Working on now (declared %d min ago): %s" % ((time.time() - h[-1]["t"]) / 60, h[-1]["text"])]
            old, seen = [], {h[-1]["text"][:40].lower()}
            for r in reversed(h[:-1]):
                k = r["text"][:40].lower()
                if k not in seen:
                    seen.add(k)
                    old.append(r)
            if old:
                L += ["Declared earlier (finished or left):"]
                L += ["- [%s] %s" % (time.strftime("%H:%M", time.localtime(r["t"])), r["text"]) for r in reversed(old[:12])]
        L += body
    return L


def claim(camp, text):
    """Declare the direction `camp` is about to work on; returns what the other islands are working on now AND what
    they have already declared, tried, kept or left (the reader must not have to open the briefs to know)."""
    import fcntl
    b = _board(camp)
    with open(str(b) + ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        with open(b, "a") as f:
            f.write(json.dumps({"camp": camp, "t": time.time(), "text": _clean(text)}) + "\n")
        cur = claims(camp)
    _ledger(SE / "camp" / camp, {"type": "claim", "text": _clean(text)})
    rec = _others_record(camp)
    L = ["Declared: %s" % _clean(text), ""]
    if rec:
        role = _json(SE / "camp" / camp / "camp.json", {}).get("role")
        L += ["CHECK YOUR DIRECTION AGAINST THE RECORD OF THE OTHER PROJECTS BELOW (same training instances as yours, so "
              "their measurements hold for you). If your direction is the same idea as, or a variant of, anything a project "
              "is working on now, declared earlier, or tried and did not keep, do NOT start it: think of another direction "
              "and declare that one instead (a new declaration replaces this one)."]
        if role in ("main", "generalist"):
            L += ["What another project kept reaches you as an offer in `offers/`; an offer is the one thing from this "
                  "record that is yours to work on (declare it as \"offer <name>: ...\").",
                  "This holds for what a project is WORKING ON NOW as well: it is being measured there on your own version "
                  "and arrives as an offer if it works, so the same idea started here spends your evaluations on something "
                  "that is already being decided."]
        else:
            L += ["What another project kept is also not for you: it is handled by the main project."]
        L += rec
    else:
        L += ["No other project has declared or tried anything yet."]
    return "\n".join(L)


def _withdrawn(rows):
    """Version numbers withdrawn by the service (ledger records of type "rollback")."""
    out = set()
    for r in rows:
        if r.get("type") == "rollback":
            out.update(r.get("dropped") or [])
    return out


def brief(rows, downward=False, keep=30, rule=True):
    """From a peer's ledger: its committed steps, what it tried and did NOT keep, what it is working on now.
    downward=True: the reader is a scout and the peer is the general island (everything listed is off limits)."""
    gone = _withdrawn(rows)                         # versions withdrawn by a rollback: not kept
    committed = {r.get("hash") for r in rows if r.get("type") == "commit" and r.get("version") not in gone}
    t_last = max([r.get("t", 0) for r in rows if r.get("type") in ("commit", "carried", "rebase")] + [0])
    kept = []
    for r in rows:
        if r.get("type") == "commit" and r.get("version") not in gone:
            p = r.get("paired") or {}
            kept.append("- v%d: %s  [%+.4f%%, t = %+.2f]" % (r["version"], _clean(r.get("note")), p.get("mean", 0.0), p.get("t", 0.0)))
    seen = {}
    for r in rows:
        if r.get("type") != "eval" or r.get("hash") in committed or not (r.get("note") or "").strip():
            continue
        pr = r.get("paired") or {}
        if pr.get("mean") is None:
            continue
        seen[_clean(r["note"])[:220]] = (r.get("t", 0), r.get("suite"), pr.get("mean", 0.0), pr.get("t", 0.0), r.get("n_inst"))
    items = sorted(seen.items(), key=lambda kv: kv[1][0])
    # "being worked on" = the latest directions since the last commit whose measurement is still in their favour
    # (mean cost lower); everything else that was measured and is not committed counts as "not kept"
    now = [it for it in items if it[1][0] > t_last and it[1][2] < 0][-3:]
    dead = [it for it in items if it not in now][-keep:]
    fmt = lambda it: "- %s  [%s on %s instances: %+.4f%%, t = %+.2f]" % (it[0], it[1][1], it[1][4], it[1][2], it[1][3])   # noqa: E731
    L = []
    if kept:
        L += ["", "## Kept there (committed steps, its own words; each with the mean cost difference to its previous version)" +
              (" - for your information only" if downward else ""), ""] + kept
    if dead:
        L += ["", "## Tried there and NOT kept (its own labels; the last measurement of each)", ""] + [fmt(it) for it in dead]
    if now:
        L += ["", "## Being worked on there right now (not decided yet)", ""] + [fmt(it) for it in now]
    if not rule:
        return L
    if downward:
        L += ["", "Nothing on these lists is for you to do. The three projects work on the same training instances, so what "
              "was measured there holds for you too: what it kept is handled by the main project and needs no second test "
              "from you; what it tried and did not keep does not work; what it is working on now it is doing itself. Do not "
              "port, re-test or retry any of it, in any guise. Your role is to give the main project ideas that nobody has "
              "had: for the three projects to succeed together their ideas must differ, so choose directions that are on "
              "none of the briefs."]
    elif dead or now:
        L += ["", "That project works on the same training instances as you. What it tried and did not keep does not work: "
              "do not try it again, in any guise. What it is working on now will reach you as an offer if it succeeds: do not "
              "duplicate it."]
    return L


# ── scout side: follow the general island ───────────────────────────────────────────────────────────────────────────
def _scout_step(camp, C, peers):
    """Secondary island: one brief per other island (kept / not kept / being worked on), everything on it off limits."""
    cd = SE / "camp" / camp
    W = cd / "enh" / "work"
    mark = cd / "priv" / "sync.json"
    S = _json(mark, {})
    for other in peers:
        H = SE / "camp" / other
        hs = _json(H / "priv" / "state.json")
        if not hs:
            continue
        rows = _rows(H)
        lab = _label(other)
        main = _json(H / "camp.json", {}).get("role") in ("main", "generalist")
        key = "%s:%s:%d:%s" % (hs.get("version"), hs.get("hash"), sum(r.get("type") == "eval" for r in rows),
                               (claims(camp).get(other) or {}).get("t"))
        if S.get("brief_" + lab) == key:
            continue
        L = ["# Brief of %s (`%s`)" % ("the MAIN project" if main else "the other scout project", lab),
             "(written %s; rewritten whenever that project evaluates or commits)" % time.strftime("%H:%M"), "",
             ("The main project improves the same original solver under the same rules, on the same training instances as "
              "you. Its solver is the one that will be tested at the end. Its committed version is v%d." +
              (" (When it has moved on, the service sets your own committed version to it at the start of a session of "
               "yours and says so in `NOTICES.md`.)" if C.get("sync_main") else "")
              if main else
              "The other scout project improves the same original solver under the same rules, on the same training instances "
              "as you. Like you it keeps its own line of versions (now v%d) and offers what it commits to the main "
              "project.") % hs.get("version", 0)]
        cl = claims(camp).get(other)
        if cl:
            L += ["", "## Declared direction (what it says it is about to work on, declared %s)" % time.strftime("%H:%M", time.localtime(cl["t"])),
                  "", "- " + cl["text"]]
        L += brief(rows, downward=True)
        (W / ("REF_%s.md" % lab)).write_text("\n".join(L) + "\n")
        S["brief_" + lab] = key
    _put(mark, S)
    return []


# ── general island side: offers from the scouts ─────────────────────────────────────────────────────────────────────
OFFER_OLD_ADVICE = ("What you do with this offer is your decision: take it as it is, take it and improve on it, combine it "
                    "with your own work, or leave it.")
OFFER_ADVICE = ("Treat it as a starting point: it was tuned on the scout's line of versions, so on your version it will "
                "usually need adapting (its parameters, where and when it is applied, which part of it to keep) before it "
                "pays off. `offers/INDEX.md` gives the order in which to work on the offers.")


def _offer_items(camp):
    """All offers of the main island `camp`, in the order of work: open ones first (most negative own t first), then
    the ones it closed, then the ones it took. Returns (items, t_accept)."""
    cd = SE / "camp" / camp
    C = _json(cd / "camp.json", {})
    reg = _json(cd / "priv" / "offers.json", {})
    rows = _rows(cd)
    ta = float((C.get("v4") or {}).get("t_accept", 1.7))
    gone = _withdrawn(rows)
    left = {}
    for r in rows:
        if r.get("type") == "offer_left":
            left[r.get("offer")] = r
        elif r.get("type") == "offer_reopened":
            left.pop(r.get("offer"), None)
    src, items = {}, []
    for name, o in reg.items():
        if name.startswith("_") or not isinstance(o, dict):
            continue
        ev = [r for r in rows if r.get("type") == "eval" and name in (r.get("note") or "") and (r.get("paired") or {}).get("mean") is not None]
        cm = [r for r in rows if r.get("type") == "commit" and name in (r.get("note") or "") and r.get("version") not in gone]
        t = o.get("t_stat")
        bands = None
        sc = o.get("from")
        if sc:
            if sc not in src:
                src[sc] = _rows(SE / "camp" / sc)
            for r in src[sc]:
                if r.get("type") == "commit" and r.get("version") == o.get("version"):
                    bands = (r.get("paired") or {}).get("bands")
        items.append({"name": name, "t": 0.0 if t is None else float(t), "mean": o.get("measured") or 0.0, "ev": ev, "cm": cm,
                      "left": left.get(name), "bands": bands})
    items.sort(key=lambda it: (2 if it["cm"] else (1 if it["left"] else 0), it["t"]))
    return items, ta


def offer_queue(camp, weak=False):
    """Names of open offers (not taken, not closed) of the main island, in the order of work. Default: those whose own
    measurement passed the first gate in the scout's project (t <= -t_accept) - the main island works on these, in
    this order, before anything else. weak=True: the other open ones (second gate: effect not established there),
    which it may take up or not."""
    items, ta = _offer_items(camp)
    return [it["name"] for it in items if not it["cm"] and not it["left"] and ((it["t"] > -ta) if weak else (it["t"] <= -ta))]


def offer_index(camp):
    """Text of `offers/INDEX.md` of the main island."""
    items, ta = _offer_items(camp)
    if not items:
        return None

    def mine(it):
        if not it["ev"]:
            return "no measurement of yours yet"
        best = min(it["ev"], key=lambda r: r["paired"]["mean"])
        return "%d evaluation(s) of yours, best %+.4f%% (t %+.2f, %s)" % (
            len(it["ev"]), best["paired"]["mean"], best["paired"].get("t", 0.0), best.get("suite"))

    def there(it):
        s = "measured there %+.4f%%, t = %+.2f%s" % (it["mean"], it["t"], "" if it["t"] <= -ta else " (second gate)")
        b = it["bands"]
        if isinstance(b, dict) and b:
            s += "; by instance size there: " + ", ".join("%s: %+.3f%%" % (k.split(": ")[-1], v[1]) for k, v in sorted(b.items()))
        return s

    L = ["# Offers: order of work", "(written by the service at %s; it is rewritten when an offer arrives and after your evaluations)" % time.strftime("%H:%M"), "",
         "t = mean difference divided by its standard error, measured in the scout's project: the more negative, the "
         "better established the effect. \"By instance size\" shows where the scout measured the effect: test an offer "
         "on the instances it is meant for. An offer is closed by a commit of yours that contains it (name it in the "
         "commit note), or - when your measurements show why it cannot pay off on your version - by `python3 "
         "tools/ctl.py leave <offer> \"<that reason, with the variants you measured>\"`.", "",
         "## Open - effect established in the scout's project (passed its first gate, t <= -%.1f)" % ta, "",
         "Work on these first, in this order, before anything else: while one of them is open the service accepts no "
         "other declaration.", ""]
    op = [it for it in items if not it["cm"] and not it["left"] and it["t"] <= -ta]
    L += ["%d. `%s`  %s  |  %s" % (k + 1, it["name"], there(it), mine(it)) for k, it in enumerate(op)] or ["(none open)"]
    wk = [it for it in items if not it["cm"] and not it["left"] and it["t"] > -ta]
    if wk:
        L += ["", "## Open - effect not established there (passed only through the second gate)", "",
              "Taking one of these up is your choice. Its measured effect is within the noise of the scout's measurement, "
              "so it competes for your time with your own ideas on its merits: look at where its effect was measured and "
              "at what your version lacks. Once you take one up, treat it like any offer: optimise it for your version "
              "and close it with a commit or with `leave`.", ""]
        L += ["- `%s`  %s  |  %s" % (it["name"], there(it), mine(it)) for it in wk]
    cl = [it for it in items if it["left"] and not it["cm"]]
    if cl:
        L += ["", "## Closed by you"]
        L += ["- `%s`  %s  |  %s  |  closed %s: %s" % (it["name"], there(it), mine(it), time.strftime("%H:%M", time.localtime(it["left"].get("t", 0))),
                                                    _clean(it["left"].get("reason"))[:300]) for it in cl]
    tk = [it for it in items if it["cm"]]
    if tk:
        L += ["", "## Taken"]
        L += ["- `%s`  %s  |  in your v%s" % (it["name"], there(it), ", v".join(str(r["version"]) for r in it["cm"])) for it in tk]
    return "\n".join(L) + "\n"


def write_offer_index(camp):
    """(Re)write `offers/INDEX.md` when its content has changed."""
    cd = SE / "camp" / camp
    W = cd / "enh" / "work"
    try:
        txt = offer_index(camp)
        ip = W / "offers" / "INDEX.md"
        body = lambda s: "\n".join(s.splitlines()[2:])   # noqa: E731  (without the time stamp line)
        if txt and (not ip.exists() or body(ip.read_text()) != body(txt)):
            ip.write_text(txt)
    except Exception as e:                            # the index must never stop the offers themselves
        _ledger(cd, {"type": "island_error", "what": "offer_index: %r" % (e,)})


def _general_step(camp, C, scouts):
    cd = SE / "camp" / camp
    W = cd / "enh" / "work"
    reg_p = cd / "priv" / "offers.json"
    reg = _json(reg_p, {})
    own_v = (_json(cd / "priv" / "state.json", {}) or {}).get("version", 0)
    out = []
    for sc in scouts:
        S = SE / "camp" / sc
        rows = _rows(S)
        if not rows:
            continue
        lab = _label(sc)
        base_of = {}                                   # scout version -> general version its tree is based on
        cur = 0
        for r in rows:
            if r.get("type") == "rebase":
                cur = r.get("from_version", cur)
                base_of[r["version"]] = cur
            elif r.get("type") == "commit":
                base_of[r["version"]] = cur
        for r in rows:
            if r.get("type") != "commit":
                continue
            n = r["version"]
            name = "%s_v%d" % (lab, n)
            if name in reg:
                continue
            a, b = S / "priv" / "versions" / ("v%d" % (n - 1)) / "src", S / "priv" / "versions" / ("v%d" % n) / "src"
            if not (a.exists() and b.exists()):
                continue
            d = W / "offers" / name
            d.mkdir(parents=True, exist_ok=True)
            pr = subprocess.run(["diff", "-ruN", "v%d/src" % (n - 1), "v%d/src" % n], cwd=str(S / "priv" / "versions"),
                                capture_output=True, text=True)
            (d / "change.diff").write_text(pr.stdout)
            changed, gone = [], []                    # the changed files in full, as the scout has them (no `patch` tool needed)
            for fb in sorted(q for q in b.rglob("*") if q.is_file()):
                rel = fb.relative_to(b)
                fa = a / rel
                if not fa.exists() or fa.read_bytes() != fb.read_bytes():
                    (d / "after" / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(fb, d / "after" / rel)
                    changed.append(str(rel))
            for fa in sorted(q for q in a.rglob("*") if q.is_file()):
                if not (b / fa.relative_to(a)).exists():
                    gone.append(str(fa.relative_to(a)))
            p = r.get("paired") or {}
            # what the scout's line was when it made this change: the original, or (camp option sync_main) the version
            # of this island it took over at one of its session starts, plus in both cases its own commits since then
            rb = [q for q in rows if q.get("type") == "rebase" and q.get("version", 0) < n and q.get("from") == camp]
            rb = rb[-1] if rb else None
            since = [q["version"] for q in rows if q.get("type") == "commit" and (rb["version"] if rb else 0) < q["version"] < n]
            prev_own = bool(since)
            base_txt = ("your v%d, which it took over at %s," % (rb.get("from_version", 0), time.strftime("%H:%M", time.localtime(rb.get("t", 0))))
                        if rb else "the original solver")
            (d / "note.md").write_text("\n".join([
                "# Offer %s" % name, "",
                "A change committed by the `%s` scout project, which works on the same training instances as you, on its own "
                "line of versions: its numbers are directly comparable with yours." % lab, "",
                "What it is, in its own words: %s" % _clean(r.get("note")), "",
                "Measured there against its previous version: mean cost difference %+.4f%% (standard error %.4f, t = %+.2f, "
                "seeds per instance %d). Negative = better." % (p.get("mean", 0.0), p.get("se", 0.0), p.get("t", 0.0), len(r.get("seeds") or [])), "",
                "The scout's line of versions when it made this change: %s plus its own earlier committed changes since then%s. "
                "`change.diff` shows the change alone, relative to that line; `after/` holds the %d changed file(s) in full as the "
                "scout has them (%s)%s. Bring the change into your `src/` by editing (there is no `patch` tool here): where your "
                "version of a file is still the one the scout started from, the file in `after/` can be copied over it; where "
                "your version has moved on, carry the change over by hand. %s" % (
                    base_txt,
                    (" (offers " + ", ".join("%s_v%d" % (lab, k) for k in since) + ")") if prev_own else " (none)",
                    len(changed), ", ".join(changed[:12]), ("; removed: " + ", ".join(gone)) if gone else "", OFFER_ADVICE),
                "", "Whatever you take from it goes through your own evaluations and your commit gate like any other change. Say "
                "in the evaluation note which offer it is (for example `offer %s: ...`), and in `NOTES.md` what you decided "
                "and why." % name, ""]))
            reg[name] = {"t": time.time(), "from": sc, "version": n, "hash": r.get("hash"), "own_version_then": own_v,
                         "measured": p.get("mean"), "t_stat": p.get("t")}
            with open(W / "NOTICES.md", "a") as f:
                f.write("- %s: new offer `offers/%s/` from the %s scout: %s  [measured there %+.4f%%, t = %+.2f]. See its `note.md`.\n" % (
                    time.strftime("%H:%M"), name, lab, _clean(r.get("note"))[:140], p.get("mean", 0.0), p.get("t", 0.0)))
            _ledger(cd, {"type": "offer_received", "offer": name, "from": sc, "version": n, "own_version": own_v,
                         "measured": p.get("mean"), "t_stat": p.get("t")})
            out.append((sc, "offer %s" % name))
        cl = claims(camp).get(sc)
        key = "%d:%s" % (sum(q.get("type") == "eval" for q in rows), (cl or {}).get("t"))
        if reg.get("_brief_" + lab) != key:            # the scout's own list of what did not work on its sizes
            L = ["# Brief of the `%s` scout project" % lab, "(written %s)" % time.strftime("%H:%M"), "",
                 "It works on the same training instances as you, keeps its own line of versions "
                 "and stays away from everything you and the other scout have tried. What it commits arrives in `offers/`."]
            if cl:
                L += ["", "## Declared direction (what it says it is about to work on, declared %s)" % time.strftime("%H:%M", time.localtime(cl["t"])),
                      "", "- " + cl["text"]]
            L += brief(rows, downward=False)
            (W / ("REF_%s.md" % lab)).write_text("\n".join(L) + "\n")
            reg["_brief_" + lab] = key
    _put(reg_p, reg)
    for nf in (W / "offers").glob("*/note.md"):       # notes written before the offer rule changed: same advice everywhere
        try:
            s = nf.read_text()
            if OFFER_OLD_ADVICE in s:
                nf.write_text(s.replace(OFFER_OLD_ADVICE, OFFER_ADVICE))
        except OSError:
            pass
    write_offer_index(camp)
    return out


def refresh_refs(camp):
    """One migration step for island `camp` (called by its driver). Returns what happened, for the log."""
    C = _json(SE / "camp" / camp / "camp.json", {})
    peers = C.get("peers", [])
    if not peers:
        return []
    try:
        if C.get("role") in ("secondary", "specialist"):
            return _scout_step(camp, C, peers)
        if C.get("role") in ("main", "generalist"):
            return _general_step(camp, C, peers)
    except Exception as ex:                             # a migration problem must never stop a campaign
        return [("error", repr(ex))]
    return []


def carry(camp, src_camp):
    """Start island `camp` from `src_camp`'s last committed version (v1 here); v0 stays the original."""
    from judge import tree_hash, copy_tree
    C, S = SE / "camp" / camp, SE / "camp" / src_camp
    st = _json(C / "priv" / "state.json")
    if st.get("version", 0) != 0:
        print("carry: %s already has committed version v%d, nothing done" % (camp, st["version"]))
        return 1
    sv = _json(S / "priv" / "state.json")["version"]
    if sv < 1:
        print("carry: %s has no committed version" % src_camp)
        return 1
    src = S / "priv" / "versions" / ("v%d" % sv) / "src"
    assert tree_hash(S / "priv" / "versions" / "v0" / "src") == tree_hash(C / "priv" / "versions" / "v0" / "src"), "different original solvers"
    W = C / "enh" / "work"
    copy_tree(src, C / "priv" / "versions" / "v1" / "src")
    copy_tree(src, W / "src")
    copy_tree(src, W / "committed")
    h = tree_hash(C / "priv" / "versions" / "v1" / "src")
    st.update(version=1, hash=h, last_eval=None, last_commit_attempt="v1 = starting point taken over from an earlier line of work")
    _put(C / "priv" / "state.json", st)
    notes = S / "enh" / "work" / "NOTES.md"
    if notes.exists():
        (W / "NOTES.md").write_text(
            "# NOTES\n\n(These notes were written by the earlier line of work whose committed version is this project's "
            "starting point, `committed/` = v1. Its evaluation numbers and instance names refer to ITS training set, which "
            "is not the one here; what it tried, what failed and why is still valid knowledge. Continue below.)\n\n"
            "---- inherited notes ----\n" + notes.read_text(errors="ignore") + "\n---- end of inherited notes ----\n\n# This project\n")
    _ledger(C, {"type": "carried", "version": 1, "from": src_camp, "from_version": sv, "hash": h})
    print("carry: %s starts from %s v%d (tree %s); original kept as v0" % (camp, src_camp, sv, h))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "carry":
        sys.exit(carry(a[1], a[2]))
    elif len(a) == 2 and a[0] == "brief":
        print("\n".join(brief(_rows(SE / "camp" / a[1]), downward=True)))
    elif len(a) == 2 and a[0] == "refs":
        print("migration step:", refresh_refs(a[1]) or "nothing new")
    else:
        sys.exit(__doc__)
