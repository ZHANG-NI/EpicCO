# Can you tell two sets of instances apart? — round {session}

You are working alone in this project directory (`{cwd}`). Run every command from it.

## Context
{problem_text}

Two sources produced instance files of this problem: source P and source Q. Someone claims that the two sources are
the same kind of instances — that a file from Q could just as well have come from P. Your job is to test that claim
as hard as you can by finding any measurable difference between the two sources.

- `study/P/` and `study/Q/` hold labelled files from each source ({n_study} of each).
- `exam/` holds {n_exam} unlabelled files, a shuffled mix of both sources (different files from the study ones).
All files were re-written by the same program in the same format, have neutral names, and their internal labelling
was shuffled, so file names, comments, formatting and element numbering carry no information.

## What to deliver
1. `tell.py`, used as `python3 tell.py <instance file>`: it prints one number in [0, 1], its probability that the file
   comes from source P. It must work on any file of this format, in a few seconds, using only the Python standard
   library.
2. `REASONS.md`: your decision criteria, most telling first. Each criterion is stated explicitly: the quantity you
   measure on a file, its typical values for P and for Q on the study files, the threshold and direction ("above
   X -> more likely P"), and how reliably it separates the study files (cross-validated). `tell.py` must implement
   exactly these criteria and nothing else: every decision it makes has to follow from a criterion written in
   `REASONS.md`. A submission without stated criteria is rejected. If you found no reliable difference, say so
   plainly — that is a legitimate result (then `tell.py` prints 0.5).
Then run `python3 tools/ctl.py submit "<note>"`: the service runs `tell.py` on the exam files, whose labels only it
knows, and tells you your accuracy. You may submit at most {max_submits} times per round; the exam labels are never
revealed.

## Rules
1. Judge the files only by what you can measure in them. Do NOT use anything you remember: no recollection of any
   benchmark's files, names, published data, or of how any benchmark or generator was constructed, and no guess about
   which source is "the real one". Every difference in `REASONS.md` must be a measurement you made on the files in
   `study/`, and `tell.py` may only compute statistics of its input file and compare them with thresholds or simple
   models that you fitted on `study/`.
2. `tell.py` must not contain stored instance data, hashes, fingerprints or look-up tables of files; such programs are
   rejected. Do not try to recognise individual files — only properties of the source.
3. There is no internet access; do not try to reach any network resource. Stay inside this directory.
4. Run commands in the foreground. No background jobs, no sub-agents, no scheduling tools.
5. Be honest about uncertainty: with {n_study} study files per source, a difference that is not clear on the study
   files will not hold on the exam. Do not over-fit the study files (check your rule with cross-validation on them).
6. Be thorough before you conclude "no difference". You must have examined at least: the distribution of every
   numeric field; how each depends on the instance size; relations between fields (for example between an element's
   value and its position, or between the global parameters and the size); spatial or structural patterns
   (clustering, spacing, margins, symmetry, repeated values, rounding, parity); and extreme values and counts of
   distinct values. `REASONS.md` must list every quantity you tested with its cross-validated accuracy on the study
   files, including the ones that did not separate the sources. A round that ends without this table is not a
   valid test.

## Working across rounds
Each round starts with an empty context and with NEW files from source Q (source Q may have been revised since the
last round; source P's files are the same). `NOTES.md` is your memory: read it first; it lists what you tried before
and how it scored. At the end of the round, update `NOTES.md` (features tried, cross-validated accuracy on study,
exam accuracy returned by the service, ideas not yet tried).

When you stop, end your final message with:
```json
{{"status": "continue", "exam_accuracy": 0.0, "summary": "2-3 sentences"}}
```
