#!/usr/bin/env python3
"""Send one request to the project's evaluation service and print its answer. Run from the project directory:
    python3 tools/ctl.py <command> [arguments...]
The call blocks until the answer is ready (an evaluation takes several minutes); run it in the foreground."""
import json
import os
import sys
import time

box = os.path.join(os.getcwd(), "box")
if not os.path.isdir(os.path.join(box, "req")):
    sys.exit("run this from the project directory (the one that contains box/ and tools/)")
if len(sys.argv) < 2:
    sys.argv.append("help")
rid = "%d_%d" % (time.time_ns(), os.getpid())
tmp = os.path.join(box, "req", rid + ".tmp")
with open(tmp, "w") as f:
    json.dump({"cmd": sys.argv[1], "args": sys.argv[2:]}, f)
os.replace(tmp, os.path.join(box, "req", rid + ".json"))
res = os.path.join(box, "res", rid + ".json")
t0 = time.time()
while not os.path.exists(res):
    time.sleep(1.0)
    if time.time() - t0 > 3 * 3600:
        sys.exit("no answer from the evaluation service after 3 hours")
with open(res) as f:
    d = json.load(f)
os.remove(res)
print(d.get("text", ""))
sys.exit(int(d.get("rc", 0)))
