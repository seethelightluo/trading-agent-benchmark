# Luna warmup archive

This is a recoverable snapshot of the Luna AC shared warmup before starting a
new DeepSeek warmup. The source was:

`agent-framework/AlphaCrafter/alphacrafter/sandbox/ws1/`

The `ws1/` copy includes the persisted factor JSON files, their calculation
formula/description/parameters, `factor_ensemble.json`, all Miner-generated
research scripts, the registered strategy, memory, audit log, workflow logs,
and persistent warmup state. No API credentials are included.

## Persisted factors

The six factor members and their formulas are:

| factor | formula |
|---|---|
| `peer_median_leadlag_5d` | `median({return_5d[j] for j != i})` |
| `short_term_reversal_5d` | `-(close / close.shift(5) - 1)` |
| `miner_2_risk_adjusted_momentum_20d` | `(close / lag(close,20) - 1) / (std(daily_returns,60) * sqrt(20))` |
| `miner_2_short_reversal_3d` | `-(close / lag(close, 3) - 1)` |
| `miner_3_clv_1d` | `-(2 * (close - low) / (high - low) - 1)` |
| `miner_3_reversal_5d` | `-pct_change(close, 5)` |

`ws1/workspace/factors/` contains the authoritative JSON definitions and
validation metrics; `ws1/workspace/scripts/` contains the corresponding
research/implementation scripts. The ensemble snapshot selected four active
members with `method=quality_ic_tilt`.

This archive is Luna evidence only. It must not be used as the DeepSeek
warmup factor library; the DeepSeek run is being rebuilt independently with
the paid DeepSeek accounts through sub2api Responses.
