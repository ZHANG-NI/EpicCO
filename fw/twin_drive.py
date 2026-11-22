#!/usr/bin/env python3
"""twin_drive.py <twin_dir>: phase 0. Alternates the imitator (writes the generator) and the discriminator (tries to
tell generated files from public ones) until the twin is certified or max_rounds is reached.

  no round yet / last round not passed  -> imitator session (must hand in an accepted generator = new round)
  round without discriminator answer    -> discriminator session
  round passed once                     -> re-issue a new blind exam from the same generator, discriminator again
  two passed rounds on the same version -> CERTIFIED
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

FW = Path(__file__).resolve().parent
sys.path.insert(0, str(FW))
import jail                                   # noqa: E402
from drive import run_session, fill, log      # noqa: E402
import twin as twinmod                        # noqa: E402


def main(tdir):
    td = Path(tdir).resolve()
    C = json.loads((td / "twin.json").read_text())
    model = C.get("model", "claude-opus-5-5")
    tmo = float(C.get("session_timeout_s", 2700))
    jail.write_net_files(Path(jail.SE) / "common" / "etc")
    svc = subprocess.Popen([sys.executable, str(FW / "twin.py"), str(td), "serve"],
                           stdout=open(td / "logs" / "twin.out", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
    (td / "logs" / "twin.pid").write_text(str(svc.pid))
    log(td, "twin driver start, service pid %d, model %s" % (svc.pid, model))
    sp = td / "priv" / "tdrive.json"
    st = json.loads(sp.read_text()) if sp.exists() else {"g": 0, "d": 0, "d_round": 0, "fails": 0}

    def state():
        return twinmod.Twin(td)

    def session(who, tpl, n, extra):
        kw = dict(C["prompt"])
        kw.update(session=n, cwd=jail.REAL_HOME + "/work", n_train=C["n_train"], lo=C.get("lo"), hi=C.get("hi"), **extra)
        tag = "%s_%03d" % (who, n)
        fin = threading.Event()
        k0 = state().st["round"]
        s0 = (state().st["rounds"][-1]["submits"] if state().st["rounds"] else 0)

        def watcher():
            # the agent's part of the round is over once its hand-in has been processed (imitator: a new round was
            # issued; discriminator: all its submissions are used): give it 3 minutes to write its notes, then stop it
            while not fin.wait(15):
                T = state()
                over = (T.st["round"] > k0 or (C.get("naive") and T.st["accepted"] >= 1)) if who == "gen" else bool(T.st["rounds"] and T.st["rounds"][-1]["submits"] >= T.max_submits
                                                                    and T.st["rounds"][-1]["submits"] > s0)
                if over:
                    if not fin.wait(180):
                        try:
                            os.killpg(int((td / "logs" / (tag + ".pid")).read_text()), 9)
                        except (OSError, ValueError):
                            pass
                    return

        threading.Thread(target=watcher, daemon=True).start()
        r = run_session(td, who, fill((FW / "prompts" / tpl).read_text(), **kw), tmo, tag, model)
        fin.set()
        log(td, "%s session %d end %.0fs rc=%s timed=%s quota=%s auth=%s" % (who, n, r["elapsed"], r["rc"], r["timed"], r["quota"], r["auth"]))
        bad = r["quota"] or r["auth"] or (r["elapsed"] < 120 and r["rc"] != 0)
        if bad:
            st["fails"] += 1
            time.sleep(300)
        else:
            st["fails"] = 0
        sp.write_text(json.dumps(st))
        return not bad

    verdict = "not certified"
    while st["fails"] < 30 and not (td / "STOP").exists():
        if svc.poll() is not None:
            svc = subprocess.Popen([sys.executable, str(FW / "twin.py"), str(td), "serve"],
                                   stdout=open(td / "logs" / "twin.out", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
        T = state()
        rounds = T.st["rounds"]
        if C.get("naive"):                       # generator written without the benchmark: no reviewer, one accepted version
            if T.st["accepted"] >= 1:
                verdict = "NAIVE GENERATOR READY"
                break
            st["g"] += 1
            if st["g"] > 4:
                break
            log(td, "naive generator session %d start" % st["g"])
            session("gen", "naive.md", st["g"], {})
            continue
        if T.certified():
            verdict = "CERTIFIED"
            break
        if not rounds:
            st["g"] += 1
            log(td, "imitator session %d start (first generator)" % st["g"])
            session("gen", "imitate.md", st["g"], {})
            continue
        rd = rounds[-1]
        dr = st.setdefault("dr", {})
        if rd["submits"] == 0 and dr.get(str(rd["round"]), 0) < 2:      # up to two discriminator sessions per round
            st["d"] += 1
            dr[str(rd["round"])] = dr.get(str(rd["round"]), 0) + 1
            log(td, "discriminator session %d start (round %d, generator v%d)" % (st["d"], rd["round"], rd["gen_version"]))
            session("dis", "discriminate.md", st["d"], {"n_study": len(T.split["study"]), "n_exam": rd["n_exam"],
                                                        "max_submits": T.max_submits})
            continue
        t0 = time.time()                       # wait for the behaviour check of this generator version
        while not state().st["rounds"][-1].get("behaviour") and time.time() - t0 < 3600:
            time.sleep(20)
        T = state()
        rd = T.st["rounds"][-1]
        log(td, "round %d: gen v%d acc %s old-programs %s behaviour %s -> %s" % (
            rd["round"], rd["gen_version"], rd["best_acc"], [round(o["acc"], 2) for o in rd["old_tells"]],
            (rd.get("behaviour") or {}).get("ok"), "pass" if T.round_passed(rd) else "not passed"))
        if T.certified():
            verdict = "CERTIFIED"
            break
        if len(T.st["rounds"]) >= int(C.get("max_rounds", 5)) + 1:
            break
        if T.round_passed(rd):                 # confirm on a fresh blind exam from the same generator
            k = rd["round"]
            (td / "cmd" / ("reissue_%d" % k)).write_text("")
            while state().st["round"] == k:
                time.sleep(5)
            continue
        st["g"] += 1
        k = rd["round"]
        log(td, "imitator session %d start (revise after round %d)" % (st["g"], k))
        session("gen", "imitate.md", st["g"], {})
        if state().st["round"] == k:           # no accepted generator came out of that session: try once more, then stop
            st["g"] += 1
            session("gen", "imitate.md", st["g"], {})
            if state().st["round"] == k:
                log(td, "imitator produced no accepted generator twice: stopping")
                break
    T = state()
    (td / "DONE").write_text(json.dumps({"verdict": verdict, "rounds": len(T.st["rounds"]), "accepted": T.st["accepted"]}))
    log(td, "TWIN_DONE %s after %d rounds (generator v%d)\n%s" % (verdict, len(T.st["rounds"]), T.st["accepted"], T.status()))
    (td / "STOP").write_text("done\n")
    time.sleep(4)
    try:
        svc.terminate()
    except OSError:
        pass


if __name__ == "__main__":
    main(sys.argv[1])
