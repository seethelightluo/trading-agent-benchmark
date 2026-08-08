# FM acceleration — Windows bundle for FM WL4–WL9

This directory is a **portable, FM-only** execution bundle intended for the
Windows U9 275HX machine. It runs six independent FactorMiner worldlines with
three long-lived workers:

```text
worker-1: WL4 → WL7
worker-2: WL5 → WL8
worker-3: WL6 → WL9
```

Each worker resumes its own persistent state after a reboot, API fault, or
process failure. A worldline is not marked complete until the entire forward
horizon completes; only then does its worker advance to the next assigned WL.

## What `bundle/` contains

- exact `scheduler/run_pipeline.py` and `FactorMiner` source matching the
  validated shared warm-up contract;
- `ASSETS.yaml` and the minimal AlphaCrafter config that FM's shared launcher
  reads even in `--mode fm`;
- WL4–WL9 full market panels;
- the completed shared FM warm-up with performance-equivalent fingerprint
  `8410ae8bbd86fd8735de5ea4823e4924cebf977e51e2946854378fba46018c28`; and
- the persisted audit certificate for the lossless source bridge
  `b93cbb67ae2e48c9 -> 8410ae8bbd86fd87`.

The script verifies that WL4–WL9 all produce the target fingerprint before any
LLM request is allowed. **Do not edit files under `bundle/` on Windows**: that
would invalidate warm-up reuse and force a new warm-up contract.

## 2026-08-03 P0/P1 performance-equivalent update (no research/config change)

This bundle now contains the validated P0/P1 FactorMiner performance changes:
P0 skips an identity daily singleton `groupby/ffill` preprocessing pass while
retaining the legacy intraday fallback; P1 evaluates independent candidate
factors in a deterministic process pool (Windows `spawn`), while the parent
process remains the only writer of library, memory, checkpoint, and admission
state.  The market panels, `ASSETS.yaml`, admission thresholds, cadence,
`iterations=200`, `target=110`, `batch=40`, online cadence, and active-factor
limit are unchanged.

A source-byte change necessarily changes the warm-up fingerprint.  To resume
existing WL4–WL9 state without re-warmup or lost windows, `fm_worker.ps1`
runs an **offline, idempotent** preflight before every scheduler start.  It
checks the shipped performance-equivalence certificate, copies each affected
historic `window_state.json` into `runtime/state/performance_equivalence_backups/`,
and changes only its certified `warmup_fingerprint` field.  It does **not**
change market data, factor libraries, signals, memory, checkpoints, completed
combination/forward outputs, scheduler state, or `runtime/.env`.  The scheduler
then performs its own explicit `--fm-performance-equivalent-from` seed/state
bridge.  The legacy `b93...` warm-up is retained read-only as the audited source
side; the new `841...` stage is the only stage verified/used for future starts.

When updating an already-copied Windows bundle, copy/overwrite **only**
`bundle/`, `scripts/`, and the top-level PowerShell/manifest/docs files.  Do
not overwrite `runtime/`: it contains the Windows machine's credentials, PIDs,
logs, backups, and per-WL resume state.

## Security

`runtime/.env` is intentionally ignored. It holds only `OPENAI_API_URL` and
`OPENAI_API_KEY`; do not commit it, archive it to a shared location, or paste
it into a terminal transcript. The Linux builder creates it from the already
configured live credentials without printing the secret. If it is absent after
transfer, create it from `runtime/.env.template` locally on the Windows PC.

## One-command start (recommended)

Double-click `RUN_FM_WL4_9.cmd`, or in PowerShell run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\RUN_FM_WL4_9.ps1
```

It creates/synchronizes the uv environment, performs the offline warm-up/data
verification, and launches the three detached workers.

## Windows first-time setup (manual equivalent)

Open **PowerShell** in this directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

`setup.ps1` requires `uv` in PATH, creates the FactorMiner uv environment,
installs the lockfile-pinned dependencies, and verifies the data/warm-up
fingerprint. It never calls the LLM API.

## Start / monitor / stop

```powershell
.\start.ps1
.\status.ps1
.\stop.ps1
```

`start.ps1` detaches three hidden PowerShell worker processes, writes their
PIDs to `runtime/pids/`, and logs them under `runtime/logs/`. Closing the
launching terminal does not terminate the workers. After a Windows reboot,
run `start.ps1` again; the per-WL state and FM checkpoints resume safely.

## Return and merge results on the Linux host

Only export after all six workers report `DONE` and after `stop.ps1` confirms
no worker remains:

```powershell
.\export_results.ps1
```

Copy the generated zip from `runtime/exports/` to the Linux repository, then:

```bash
python3 'FM acceleration/scripts/merge_results.py' \
  --archive /path/to/FM_WL4_9_results_*.zip
```

The merge tool validates the warm-up fingerprint, refuses to overwrite a
non-empty WL4–WL9 destination, copies the WL artifacts, and rewrites portable
Windows absolute paths inside saved JSON state to this repository's paths.
