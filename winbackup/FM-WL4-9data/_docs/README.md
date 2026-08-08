# FM acceleration — Windows bundle for FM WL4–WL9　（暂停：`.\stop.ps1` ｜ 重新启动：`powershell -ExecutionPolicy Bypass -File ".\RUN_FM_WL4_9.ps1"`）

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
- the completed shared FM warm-up with fingerprint
  `b93cbb67ae2e48c9be026297cee2fe40fdbfb2cf5cbfa03c5d6bf89376964b3c`.

The script verifies that WL4–WL9 all produce that same fingerprint before any
LLM request is allowed. **Do not edit files under `bundle/` on Windows**: that
would invalidate warm-up reuse and force a new warm-up contract.

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
