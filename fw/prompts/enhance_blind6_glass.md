# Improve a state-of-the-art solver — session {session}

You are working alone in this project directory (`{cwd}`). Run every command from it.

## The problem and the solver
{problem_text}

`src/` holds the solver you work on. It starts as the unmodified original ({solver_text}); `orig/` keeps an untouched
copy of the original and `committed/` the last committed version, both for reference (do not edit them).{ref_text}

## The goal
Make this solver a better general solver for this problem, under a fixed per-instance time limit: {time_text}, on one
CPU core.

You work with the training set in `instances/train/`: {n_train} FAMILIES of order books of several kinds (from a few
dozen to several hundred items; long stacks, single items, two stacks only, many or few repeated sizes). A file of
the training set is a recipe, not one order book: THE SEED OF A RUN SELECTS THE ORDER BOOK of the family (the same
family and seed always give the same order book; `python3 tools/problem.py make <file> <seed> <dir>` writes its
three csv files). No optimal or best-known values exist: `targets.json` lists, for each family, the smallest waste
that any version has reached on any of its order books, and the gap of a run is 100 x (cost - reference) /
reference. Since the order books of a family differ in size, a gap by itself says little: what is measured is the
DIFFERENCE between two versions on the same order books.{rotate_text}{island_text}
Every version is measured by the project's evaluation service, never by the solver's own report: the solution file
is re-read, checked for feasibility and its cost recomputed.

The training set is only a sample. After the project your last committed version is tested, against the original and
against other published solvers, on order books you have never seen (real ones, each run once). A change that only helps
on the files in this directory is worth nothing; a change that makes the solver better at this problem is the goal.

## Tools
- `python3 tools/ctl.py status` — committed version, whether `src/` differs from it, evaluations so far.
- `python3 tools/ctl.py run <instance> [<instance> ...] [--seed k]` — build `src/` and run it once per listed instance
  (order book k of each family),
  each on a whole core under the official time limit, in parallel (up to 32 instances per call; instance = file name
  without extension, from `instances/train/`). Prints the verified cost and gap of each run; the solver's output is
  copied to `runs/single/`. With `--long` (at most 8 instances per call) the run gets 5 times the time limit: it
  shows what this solver reaches when time is not binding, and its result becomes the instance's reference when it
  is better. A long run is a diagnostic, never a result: versions are compared under the official time limit only.
- `python3 tools/ctl.py probe <instance> <instance> ... "<note>"` — your quick test: a few instances of the training
  set that YOU choose (2 to 8), several seeds (order books) each, compared with the committed version on the same
  order books. It answers in about seven to thirteen minutes (every run uses the whole time limit). A probe has its
  own seeds: `eval full` measures a tree again on other order books, so a figure that was the best of several probes
  will usually be smaller there. Use it for most of your trials, choose the families
  that represent what you are working on, and keep the choice fixed while you compare variants of one idea.
- `python3 tools/ctl.py eval full "<note>"` — the official measurement: every training family, several seeds (order
  books) each, every run on its own core, compared with the committed version and with the original. When the result is
  promising but not conclusive it is re-measured on more seeds automatically. Takes roughly {eval_minutes} minutes
  per round of seeds. Results are written to `runs/eval_<n>/` (`summary.md`, `per_instance.tsv`, `logs/`).
- `python3 tools/ctl.py commit "<note>"` — make `src/` the new committed version. Accepted only after a complete
  `eval full` of exactly this tree, and only if it passes the commit gate: {blind_gate_text} A refused commit changes
  nothing; read the answer.
- `python3 tools/ctl.py revert` — restore `src/` from the committed version.
You can compile the solver yourself in this directory (`g++`, `python3` are available;
`python3 tools/problem.py check src` reports the compile errors of `src/` within seconds, and
`python3 tools/problem.py build src bin` builds `bin/solver` exactly as the official build does) and run it
for a few seconds to check that it builds and does not crash. That is all this shell is for: it has two slow shared
cores, and nothing measured here is evidence. Every experiment that decides anything — a parameter comparison as
much as a new mechanism — goes through `ctl.py run` or `ctl.py eval`, which use dedicated cores under the official
time limit. Official builds use exactly: {build_text}.

## How the measurements work — read this before planning
- This solver has no random number: the same code on the same order book under the same time limit returns the same
  waste (except where the time limit happens to cut the search a moment earlier or later). What varies is the order
  book, so every comparison is a statistical one over order books. Each evaluation reports the mean cost
  difference to the committed version (per order book, in percent, averaged per family, then over the families),
  its standard error and t = difference / standard error. |t| below about 1 is noise; do not draw conclusions from
  it, and do not read anything into a single order book. With several order books per family you can see, family
  by family, whether a difference holds from one order book to the next.
- Your tree and the committed version are always run on the SAME order books, and after every commit a new set of
  seeds - new order books - is used: a version is never judged twice on the order books that won its commit. A
  tree identical to the committed version scores a difference of zero, and a change that leaves the search itself
  unchanged (a pure speed-up) is measured precisely: it reaches the same solutions earlier and goes on from there.
- Evaluations are the scarce resource of this project, far scarcer than your own time: the compute behind them is
  fixed. Spend them on questions whose answer changes what you do next. A probe answers "is this idea alive?"; a
  full evaluation answers "can this be committed?". Tuning a numeric parameter by a series of full evaluations
  wastes most of the project.
- Change one thing per evaluation. A tree that bundles several untested changes cannot tell you which of them
  helped and which hurt, and one bad change sinks the good ones with it.
- Small improvements are welcome: the gate re-measures a promising tree on more seeds, so a modest but real gain
  passes on its own. Commit each verified gain separately and build the next change on top of it. A change that
  makes the solver do the same work faster is such a gain (under a fixed time limit, speed is quality).
- The reference values in `targets.json` are only the best costs found so far by this solver family under this
  time limit; they are not optima. A small gap to the reference does not mean the solver is close to optimal.
- Think about where the remaining loss comes from before choosing what to change (the solver logs of an evaluation
  show how each run progressed over time), and prefer changes of mechanism over changes of constants.
- The final test weighs every order book equally, the small ones as much as the large ones. Two different
  situations cost quality, and they need different remedies. On many order books the search is still improving
  when the time is up: anything that makes it reach good solutions earlier helps. On others it stops improving
  long before the time is up, at a solution that is NOT the best one: there, speed buys nothing, and what is missing
  is search quality (which partial solutions are kept, which are expanded first, what is cut off). A long run
  (`ctl.py run --long`) shows which situation an order book is in. Do not conclude that a family has nothing to
  gain because all versions tie on it, and improvements on the largest order books alone will not make this a
  much better solver.
- The training set is small, and the final test is not: it has many instances of every size, with other
  structures (how many items, how their sizes repeat, how long the stacks are, where the defects lie). A change
  that works through a property of these particular files will not survive it. Prefer mechanisms whose reason to
  work does not depend on the instance, and say in `NOTES.md` why you expect each committed change to carry over.

## Rules
1. Stay inside this directory. There is no internet access; do not try to reach any network resource.
2. It must remain this solver: keep its core method as the search engine and improve it from the inside (any change
   to its components, data structures, operators, control logic or parameters is allowed, as are additional
   components). Replacing it with a different algorithm written from scratch is not what this project studies.
3. One program for all instances: no branching on instance names or on recognised files, no stored solutions or
   instance-specific tables, no reading of anything except the instance's files given on the command line.
4. One instance, one core: the solver runs as a single thread in a single process. No threads, no parallel
   libraries, no child processes, no change of CPU affinity — a source tree that contains any of these is refused
   at build time, and a run whose CPU time exceeds its wall-clock time is counted as failed. `src/host_main.cpp`
   (the program's arguments, the choice of the search from `params.txt`, its output) and the two helper headers
   `src/benchtools/info.hpp` and `src/nlohmann/json.hpp` are not part of the solver and must stay exactly as in
   `orig/`: a tree in which one of them differs is refused at build time.
5. Keep the command line and the outputs unchanged ({io_text}) and respect the time limit: a run that exceeds it is
   counted as failed.
6. Run commands in the foreground. No background jobs, no sub-agents, no scheduling tools. Do not run the solver
   yourself for more than a few seconds, and never several copies at once: experiments go through `ctl.py`.
7. Report only numbers you observed in files of this directory.

## Working across sessions
The project runs as a sequence of sessions; each starts with an empty context. `NOTES.md` is your memory: read it
first, then run `python3 tools/ctl.py status`, and read `NOTICES.md` if it exists. Keep `NOTES.md` up to date as you
go (what you changed and why, what each evaluation showed with its eval id, what is committed, what failed and should
not be retried, what to try next) — a session can end at any moment, so write the notes right after every evaluation,
not at the end.

Current status:
```
{status}
```

When you stop, end your final message with:
```json
{{"status": "continue" | "done", "committed_version": 0, "summary": "2-3 sentences"}}
```
Use "done" only if you have nothing left worth trying.
