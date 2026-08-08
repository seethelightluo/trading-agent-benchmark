# P0/P1 performance-equivalent Windows bundle update — 2026-08-03

This is a **code-only, lossless** update for the portable Windows FM WL4–WL9
bundle.  No experiment was started while preparing it.  No market panel,
worldline, FM/portfolio/admission setting, API credential, runtime state, PID,
log, or output was deleted or overwritten.

## Unchanged experimental contract

- FM cadence: 10 trading days
- shared warm-up: 200 iterations, target 110, batch 40
- online mining: one iteration per window
- evaluation workers: 4
- IC / ICIR / correlation and portfolio settings: unchanged from `ASSETS.yaml`
- shared warm-up profile: `live_cap1000000_atomicxfer1_ic0p007_ir0p084_rho0p5_i200_t110_b40`

## Included performance-only changes

- **P0**: daily singleton preprocessing avoids a provably identity grouped
  forward-fill; intraday data keeps the legacy path.
- **P1**: candidate evaluation uses a deterministic process pool.  On Windows
  it uses `spawn`; only the parent writes persistent FM state.

The target implementation is certified performance-equivalent to the old
research contract:

```text
source warm-up: b93cbb67ae2e48c9be026297cee2fe40fdbfb2cf5cbfa03c5d6bf89376964b3c
target warm-up: 8410ae8bbd86fd8735de5ea4823e4924cebf977e51e2946854378fba46018c28
```

The certificate is shipped under:

```text
bundle/agent-framework/results/fm/performance_equivalence/
```

## Existing Windows checkpoints: lossless window handling

Before the first new-code scheduler launch for each WL, `fm_worker.ps1` runs
`scripts/migrate_window_states.py`.  It validates the certificate, backs up
all affected historic `window_state.json` files under
`runtime/state/performance_equivalence_backups/`, and retags only their
certified fingerprint.  It leaves scheduler state and seed manifests untouched
so `run_pipeline --fm-performance-equivalent-from b93cbb67ae2e48c9be026297cee2fe40fdbfb2cf5cbfa03c5d6bf89376964b3c` can perform its
normal audited seed/state migration.  It never changes factors, signals,
memory, checkpoints, forward results, worldline data, or credentials.

## Transfer rule

Copy the updated **bundle, scripts, top-level PowerShell files, manifest, and
this note** to the Windows PC.  Do **not** copy or overwrite `runtime/`; it is
machine-local persistent state and includes `.env`.
