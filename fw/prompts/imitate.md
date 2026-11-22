# Fresh instances of a public benchmark's kind — round {session}

You are working alone in this project directory (`{cwd}`). Run every command from it.

## Context
{problem_text}

`public/` holds a public benchmark of this problem ({n_train} instance files). A solver that is tuned on a fixed set
of files can end up fitted to those particular files. To prevent that, NEW, ORDINARY instances of the same kind are
needed — instances that could have been one more file of this benchmark. Your job is to write the generator that
produces them.

## What to produce
`gen/gen.py`, an instance generator with exactly this interface:
    python3 gen/gen.py --family <name> --n <size> --seed <int> --out <path>
It writes ONE instance file in the public benchmark's format to `<path>`, deterministically for a given
(family, n, seed). {size_text}
`gen/families.json`: a list of `{{"name": "<alphanumeric id>", "description": "<one sentence>"}}`, one entry per
family that `gen.py` supports.

A family is one of the structural variants that the public benchmark ITSELF contains. Work out the benchmark's own
recipe from the files in `public/` — measure their features, look at how the numbers are distributed and how they
depend on each other and on the size — and reproduce it: the same variants, in the same proportions and parameter
ranges, with new random draws.
- Typical instances only ({class_text}). Do NOT invent families the public benchmark does not contain: no unusual
  structure, no extreme parameter values, no boundary or special cases.
- NEVER copy: the generator must not read `public/` (it is not available when the generator runs), must not contain
  data taken from public files (no embedded tables), and must not output public files or modified versions of them.
  Every generated instance is compared with every public file; copies are rejected, and so are generators that store
  data.
- Instances whose basic features fall outside the range seen in the public benchmark are rejected.
Use only the Python standard library.

## How your generator is checked
`python3 tools/ctl.py submit "<note>"` hands in the current `gen/`. The service then draws, with its own seeds, one
instance for each size that occurs in the public benchmark, validates them, and has them examined in two ways:
1. A reviewer is shown labelled public files and labelled files of yours, and must then classify unseen files as
   public or yours. If it succeeds clearly more often than chance, your instances are not yet of the same kind; the
   reviewer's stated criteria (what it measured and which values gave your files away) are written to
   `feedback/latest.md`.
2. A reference solver is run on public files and on your files of the same sizes; its behaviour should be the same.
The reviewer only starts after your round has ended, so do NOT wait for it: once a `submit` has been accepted, update
`NOTES.md` and finish. Its result will be in `feedback/latest.md` at the start of your next round
(`python3 tools/ctl.py feedback` prints the same file).
`python3 tools/problem.py features <file>` prints the basic features of an instance file.
You may run `gen/gen.py` yourself to inspect what it produces (write test files under `scratch/`).

## How to use the feedback
Each criterion the reviewer reports is a measurable way in which your files differ from the public ones. Measure that
quantity yourself on `public/` and on your own output, find out which part of your recipe causes the difference, and
correct the recipe so that the quantity is distributed as in the public files. Fix causes, not symptoms: do not add
special-case patches that only move one statistic, and never get closer to the public files by reusing their data.

## Rules
1. Stay inside this directory. There is no internet access; do not try to reach any network resource.
2. Work from the files in `public/` and your own measurements.
3. Run commands in the foreground. No background jobs, no sub-agents, no scheduling tools.

## Working across rounds
Each round starts with an empty context. `NOTES.md` is your memory: read it first, then `feedback/latest.md` if it
exists. In each round: address the reviewer's criteria, test the generator, `submit` once (twice at most if the first
attempt is rejected), and update `NOTES.md` (the recipe as you currently understand it, what the reviewer found, what
you changed, what is still unexplained).

When you stop, end your final message with:
```json
{{"status": "continue", "families": 0, "summary": "2-3 sentences"}}
```
