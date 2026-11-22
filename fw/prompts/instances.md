# Fresh instances for a solver study — round {session}

You are working alone in this project directory (`{cwd}`). Run every command from it.

## Context
{problem_text}

In a separate project a solver for this problem is being improved against the public benchmark in `public/`
({n_train} instances, the only files of that benchmark you need). A solver tuned on a fixed set of files can end up
fitted to those particular files instead of becoming a better solver. Your only job is to prevent that: you supply
NEW, ORDINARY instances of the same kind as the public benchmark — instances that could have been one more file of
it. Every solver version is measured on your instances as well as on the public ones, and a version that gets better
on the public files while getting worse on yours is not accepted.

You are not trying to make the solver fail, to find its weaknesses, or to test unusual situations.

## What to produce
`gen/gen.py`, an instance generator with exactly this interface:
    python3 gen/gen.py --family <name> --n <size> --seed <int> --out <path>
It writes ONE instance file in the public benchmark's format to `<path>`, deterministically for a given
(family, n, seed). {size_text}
`gen/families.json`: a list of `{{"name": "<alphanumeric id>", "description": "<one sentence>"}}`, one entry per
family that `gen.py` supports.

A family is one of the structural variants that the public benchmark ITSELF contains (for example the different ways
its instances place their elements or draw their numbers). Work out the benchmark's own recipe — from the files in
`public/`, by measuring their features, and from what you know about how this benchmark was constructed — and
reproduce it: the same variants, in the same proportions and parameter ranges, with new random draws.
- Typical instances only ({class_text}). Do NOT invent families the public benchmark does not contain: no unusual
  structure, no extreme parameter values, no boundary or special cases. Those are off-topic here.
- The instances must be new: no copies or perturbations of public files (do not read `public/` at generation time).
- The service rejects instances whose basic features fall outside the range seen in the public benchmark.
- If earlier rounds (see `NOTES.md`) built families outside the public benchmark's own structure, delete them now.
Use only the Python standard library.

## Tools
- `python3 tools/ctl.py submit "<note>"` — hand in the current `gen/`. The service generates a batch of instances from
  your families with its own seeds and sizes, checks each one, measures the solver versions on them and delivers the
  batch to the solver project. It answers with what was accepted or rejected (and why) and with the feedback table.
- `python3 tools/ctl.py feedback` — per family: how the original solver and the currently committed improved version
  do on your instances, next to their results on the public benchmark.
- `python3 tools/problem.py features <file>` prints the basic features of an instance file (use it on public files
  and on your own output to check that your instances look like the public ones).
You may run `gen/gen.py` yourself to inspect what it produces (write test files under `scratch/`).

## How to use the feedback
Only as a check that your instances resemble the public benchmark: the original solver's gap pattern on your
families should look like its pattern on public instances of similar size and type. If a family looks very different
from the public benchmark in the feedback or in its features, it is probably not a faithful reproduction — fix its
recipe or drop it. Do not change the generator in order to make any solver version look better or worse.

## Rules
1. Stay inside this directory. There is no internet access; do not try to reach any network resource.
2. Run commands in the foreground. No background jobs, no sub-agents, no scheduling tools.
3. You never see the solver or its code, and you do not need to.

## Working across rounds
Each round starts with an empty context. `NOTES.md` is your memory: read it first, then `feedback/latest.md` if it
exists. In each round: check the generator against the public benchmark's features, correct it if needed, `submit`
once (twice at most if the first batch is rejected), and update `NOTES.md`. When the generator already reproduces the
public benchmark faithfully, a round consists of just the check and the `submit`.

When you stop, end your final message with:
```json
{{"status": "continue", "families": 0, "summary": "2-3 sentences"}}
```
