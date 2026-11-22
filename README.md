# EpicCO

Framework of the paper **"EpicCO: Evolving Frontier Solvers for Combinatorial Optimization with LLM Agents"**.

## Enhanced solvers

Each solver produced by the framework has its own repository:

| Problem | Host solver | Repository |
|---|---|---|
| Capacitated Vehicle Routing | HGS-CVRP | [EpicCO-CVRP](https://github.com/ZHANG-NI/EpicCO-CVRP) |
| Job-Shop Scheduling | the tabu search of Li, Hao and Wu (HTSFP) | [EpicCO-JSSP](https://github.com/ZHANG-NI/EpicCO-JSSP) |
| Two-Dimensional Guillotine Bin Packing | GDRR | [EpicCO-BinPacking](https://github.com/ZHANG-NI/EpicCO-BinPacking) |
| Weighted Vertex Coloring | MCTS + RedLS | [EpicCO-VertexColoring](https://github.com/ZHANG-NI/EpicCO-VertexColoring) |
| Maximum-Weight Independent Set | CHILS | [EpicCO-IndependentSet](https://github.com/ZHANG-NI/EpicCO-IndependentSet) |
| Critical Node Detection | IRMS | [EpicCO-CriticalNodes](https://github.com/ZHANG-NI/EpicCO-CriticalNodes) |
| Glass Cutting | MBA* | [EpicCO-GlassCutting](https://github.com/ZHANG-NI/EpicCO-GlassCutting) |
| Last-Mile Routing | the first-ranked approach of the challenge | [EpicCO-LastMileRouting](https://github.com/ZHANG-NI/EpicCO-LastMileRouting) |

## Requirements

- Linux (x86-64), Python 3 (standard library only)
- [proot](https://proot-me.github.io/) for the sandbox, Slurm for the job scripts
- the compilers of the hosts: g++/gcc; a Rust toolchain for the bin-packing host
- the Claude Code command-line client; the campaigns of the paper used the model `claude-opus-5-5`. The repository
  contains no API key and no credential: the sessions use the login of the client installed on the machine (its
  location can be set with `SE_CRED`), which is made visible inside the sandbox and nowhere else

## Running a campaign

The scripts expect the repository at `$SE_ROOT` (default `/opt/epicco`); this path also appears in the task files
and is replaced by `$SE_ROOT` when a campaign is created.

1. Edit `cluster/env.sh` and the `#SBATCH` lines of `cluster/*.sbatch` (`<partition>`, `<qos>`) for your cluster.
2. Put the starting version of the solver (the host's source, with the command-line interface that
   `fw/problems/<problem>.py` describes) where the task file expects it (`upstream` in `fw/tasks/<problem>_*.json`).
3. Create the development instances with the generator of the problem and the evaluation tools the plug-in
   `fw/problems/<problem>.py` names.
4. Start one job per agent, for example for the three agents of the vehicle-routing campaign:

   ```
   SE_PLAIN=1 SE_TASK=tasks/cvrp_isl3_G.json  sbatch -J lead   cluster/arm.sbatch cvrp_G3
   SE_PLAIN=1 SE_TASK=tasks/cvrp_isl3_S5.json sbatch -J scout1 cluster/arm.sbatch cvrp_S5
   SE_PLAIN=1 SE_TASK=tasks/cvrp_isl3_S6.json sbatch -J scout2 cluster/arm.sbatch cvrp_S6
   ```

   The header of `cluster/arm.sbatch` lists the options. Every evaluation, accepted version and migration is
   recorded in `camp/<name>/priv/ledger.jsonl`, the accepted versions in `camp/<name>/priv/versions/`.
5. Final test: `python3 fw/final_eval.py <spec.json>` (format in the file's header), then `fw/final_tables.py`.
