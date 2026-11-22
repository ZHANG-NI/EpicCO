"""proot jail used for every LLM session and for running LLM-written generator code.

Inside the jail /data and /home are empty; the only home is the session's own directory. Name resolution is cut
down to an allow-list (the model API hosts) through a private /etc/hosts + a dead resolver, shell children get dead
proxy variables, and the web tools are denied on the command line: the model has no internet access.
/proc is not isolated (proot cannot); transcripts are audited afterwards.
"""
import fcntl
import glob
import json
import os
import socket
import time
from pathlib import Path

# Site configuration through the environment; the defaults are those of the first machine. On another machine (cluster/env.sh):
#   SE_ROOT          project root                     SE_REAL_HOME   home directory as seen INSIDE the jail
#   SE_TOOLCHAIN     directory of the compiler (host) SE_TOOLCHAIN_IN  where it appears inside the jail
#   SE_HIDE          colon list of directories replaced by an empty one inside the jail
#   SE_JAIL_PATH / SE_JAIL_LD   extra PATH / LD_LIBRARY_PATH inside the jail (compiler modules)
#   SE_SESSION_CPUS  CPUs for agent sessions (inside a Slurm job that also holds the evaluation pool)
SE = os.environ.get("SE_ROOT", "/opt/alphaco")
REAL_HOME = os.environ.get("SE_REAL_HOME", "/home/agent")
HOST_HOME = os.path.expanduser("~")                 # where the real credentials file lives (unless SE_CRED names it)
CRED_FILE = os.environ.get("SE_CRED") or (HOST_HOME + "/.claude/.credentials.json")
TOOLCHAIN_REAL = os.environ.get("SE_TOOLCHAIN", "/opt/alphaco/toolchain")
TOOLCHAIN_IN = os.environ.get("SE_TOOLCHAIN_IN", REAL_HOME + "/toolchain_env")
HIDE = [d for d in os.environ.get("SE_HIDE", "/data:/home").split(":") if d]
JAIL_PATH = os.environ.get("SE_JAIL_PATH", "")
JAIL_LD = os.environ.get("SE_JAIL_LD", "")
ALLOW_HOSTS = ["api.anthropic.com", "console.anthropic.com", "platform.claude.com", "claude.ai", "statsig.anthropic.com"]


def write_net_files(etc_dir):
    etc = Path(etc_dir)
    etc.mkdir(parents=True, exist_ok=True)
    lines = ["127.0.0.1 localhost"]
    for h in ALLOW_HOSTS:
        try:
            ips = sorted({i[4][0] for i in socket.getaddrinfo(h, 443, proto=socket.IPPROTO_TCP)})
        except Exception:
            ips = []
        lines += ["%s %s" % (ip, h) for ip in ips]
    (etc / "hosts").write_text("\n".join(lines) + "\n")
    (etc / "resolv.conf").write_text("nameserver 127.0.0.1\noptions timeout:1 attempts:1\n")
    ns = []
    try:
        for line in open("/etc/nsswitch.conf"):
            ns.append("hosts:          files\n" if line.strip().startswith("hosts:") else line)
    except OSError:
        ns = ["passwd: files\n", "group: files\n", "hosts: files\n"]
    (etc / "nsswitch.conf").write_text("".join(ns))
    return etc


def prepare_home(home):
    home = Path(home)
    for sub in ("work", "tmp", ".claude", "toolchain_env", "tools"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    cj = home / ".claude.json"
    if not cj.exists():
        cj.write_text('{"hasCompletedOnboarding": true}')
    (home / ".bashenv").write_text(
        "export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 "
        "HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9\n")
    cred = home / ".claude" / ".credentials.json"
    if not cred.exists():
        cred.write_text("{}")
    return home


def _cpus(s):
    out = set()
    for part in s.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def slurm_cpus():
    """CPUs that belong to some Slurm job on this node (the evaluation pool among them)."""
    used = set()
    for f in glob.glob("/sys/fs/cgroup/system.slice/*slurmstepd*/job_*/cpuset.cpus.effective"):
        try:
            used |= _cpus(open(f).read())
        except OSError:
            pass
    return used


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def alloc_cpus(tag, cores=2):
    """Reserve `cores` physical cores (all their hardware threads) for one agent session or service. They lie outside
    every Slurm job's cpuset - in particular outside the evaluation pool - and outside the other reservations, so
    nothing an agent runs in its jail can disturb a timed solver run. The reservation lapses when its owner dies."""
    fixed = os.environ.get("SE_SESSION_CPUS", "")
    if fixed:                                   # inside a Slurm job: the job script set the session CPUs aside
        return sorted(_cpus(fixed))
    reg = Path(SE) / "common" / "cpu_alloc.json"
    with open(str(reg) + ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            R = json.loads(reg.read_text())
        except Exception:
            R = {}
        R = {k: v for k, v in R.items() if _alive(v.get("pid"))}
        if tag not in R or R[tag].get("pid") != os.getpid():
            busy = slurm_cpus() | {c for k, v in R.items() if k != tag for c in v["cpus"]}
            phys = {}
            for c in range(os.cpu_count()):
                try:
                    sib = sorted(_cpus(open("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % c).read()))
                except OSError:
                    sib = [c]
                phys[sib[0]] = sib
            picked = []
            for key in sorted(phys, reverse=True):          # from the top of the machine: Slurm fills from the bottom
                if len(picked) >= cores:
                    break
                if not any(c in busy for c in phys[key]):
                    picked.append(phys[key])
            R[tag] = {"cpus": [c for sib in picked for c in sib], "pid": os.getpid(), "t": time.time()}
        reg.write_text(json.dumps(R))
        return R[tag]["cpus"]


def kill_session(marker):
    """Kill every process of mine whose environment carries SE_SESSION=<marker> (prefix match when marker ends with
    ':'). Agent tools start their commands in sessions of their own, so killing a process group is not enough."""
    me, n = os.getuid(), 0
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            if os.stat("/proc/" + d).st_uid != me:
                continue
            env = open("/proc/%s/environ" % d, "rb").read().split(b"\0")
        except OSError:
            continue
        for e in env:
            if e.startswith(b"SE_SESSION="):
                v = e[11:].decode("utf-8", "ignore")
                if v == marker or (marker.endswith(":") and v.startswith(marker)):
                    try:
                        os.kill(int(d), 9)
                        n += 1
                    except OSError:
                        pass
                break
    return n


def jail_cmd(home, cmd, cwd=REAL_HOME + "/work", env=None, cred=False, etc_dir=None, cpus=None):
    """argv that runs `cmd` inside the jail whose home directory is `home` (pinned to `cpus` when given)."""
    home = str(prepare_home(home))
    etc = str(etc_dir or (Path(SE) / "common" / "etc"))
    os.makedirs(SE + "/common/empty", exist_ok=True)
    a = [SE + "/common/proot"]
    for d in HIDE:
        a += ["-b", SE + "/common/empty:" + d]
    a += ["-b", home + ":" + REAL_HOME]
    if TOOLCHAIN_REAL:
        a += ["-b", TOOLCHAIN_REAL + ":" + TOOLCHAIN_IN]
    a += [
         "-b", SE + "/common/tools:" + REAL_HOME + "/tools",
         "-b", home + "/tmp:/tmp",
         "-b", etc + "/hosts:/etc/hosts",
         "-b", etc + "/nsswitch.conf:/etc/nsswitch.conf",
         "-b", etc + "/resolv.conf:/etc/resolv.conf"]
    rp = os.path.realpath("/etc/resolv.conf")
    if rp != "/etc/resolv.conf":
        a += ["-b", etc + "/resolv.conf:" + rp]
    if cred:
        a += ["-b", CRED_FILE + ":" + REAL_HOME + "/.claude/.credentials.json"]
    e = {"HOME": REAL_HOME, "TMPDIR": REAL_HOME + "/tmp", "TERM": "dumb", "LANG": "C.UTF-8",
         "PATH": REAL_HOME + "/tools:" + (TOOLCHAIN_IN + "/bin:" if TOOLCHAIN_REAL else "") + (JAIL_PATH + ":" if JAIL_PATH else "") + "/usr/local/bin:/usr/bin:/bin",
         "BASH_ENV": REAL_HOME + "/.bashenv", "PYTHONUNBUFFERED": "1"}
    if JAIL_LD:
        e["LD_LIBRARY_PATH"] = JAIL_LD
    e.update(env or {})
    a += ["-w", cwd, "env", "-i"] + ["%s=%s" % kv for kv in e.items()] + list(cmd)
    if cpus:
        a = ["taskset", "-c", ",".join(str(c) for c in cpus)] + a
    return a


def proot_env():
    e = dict(os.environ)
    e["PROOT_TMP_DIR"] = SE + "/common/proot_tmp"
    os.makedirs(e["PROOT_TMP_DIR"], exist_ok=True)
    return e
