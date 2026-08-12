"""miner1 2028-05-26: candidate B - vol-scaled momentum (risk-adjusted momentum).
Hypothesis: de-weighting high-vol names (crypto/commodities) by scaling momentum
with realized vol should avoid momentum top-pick crashes in risk-off regimes.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner1_ic_lib import load_panel, ic_series, fwd_returns, summarize_ic, coverage_stats, turnover_rank, turnover_signal

panel = load_panel()
close = panel['close']
fw = fwd_returns(close, horizons=(1, 2, 3, 5, 10))

def mom_skip(px, n, skip):
    return px.shift(skip) / px.shift(skip + n) - 1.0

def rv(px, n):
    return px.pct_change().rolling(n).std()

def sma(px, n):
    return px.rolling(n).mean()

cands = {}
# V1: 20d momentum / 20d realized vol (Sharpe-like)
cands['mom20_skip3_div_rv20'] = mom_skip(close, 20, 3) / rv(close, 20)
# V2: 60d momentum / 60d realized vol
cands['mom60_skip5_div_rv60'] = mom_skip(close, 60, 5) / rv(close, 60)
# V3: 120d momentum / 60d realized vol (existing mom uses 120d/5 skip)
cands['mom120_skip5_div_rv60'] = mom_skip(close, 120, 5) / rv(close, 60)
# V4: 120d momentum / 20d realized vol
cands['mom120_skip5_div_rv20'] = mom_skip(close, 120, 5) / rv(close, 20)
# V5: 120d momentum * (1/rv60) rank-normalized soft version
cands['mom120_skip5_div_rv20_ann'] = mom_skip(close, 120, 5) / (rv(close, 20) * np.sqrt(252))

for name, f in cands.items():
    f = f.loc['2021-01-01':]
    ic = ic_series(f, fw[1])
    s = summarize_ic(ic, name)
    cov = coverage_stats(f)
    if s:
        print(f"{name:32s} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} t={s['t_stat']:+.2f} n={s['n_dates']} cov_dates={cov['dates_valid_ge8']} avg_valid={cov['avg_valid']:.1f}")

print("\n--- decay for best variants ---")
for name in ['mom20_skip3_div_rv20', 'mom60_skip5_div_rv60', 'mom120_skip5_div_rv60']:
    f = cands[name].loc['2021-01-01':]
    print(name)
    for h in [1, 2, 3, 5, 10]:
        ic = ic_series(f, fw[h])
        s = summarize_ic(ic, f'h{h}')
        if s:
            print(f"  h={h:2d} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} n={s['n_dates']}")

print("\n--- sub-period robustness (1d IC) ---")
for name in ['mom20_skip3_div_rv20', 'mom60_skip5_div_rv60', 'mom120_skip5_div_rv60']:
    f = cands[name]
    print(name)
    for lo, hi in [('2021-01-01', '2022-12-31'), ('2023-01-01', '2024-12-31'), ('2025-01-01', '2026-07-15'), ('2026-07-16', '2027-12-31'), ('2028-01-01', '2028-06-08')]:
        ff = f.loc[lo:hi]
        ic = ic_series(ff, fw[1].loc[lo:hi])
        s = summarize_ic(ic, f'{lo}..{hi}')
        if s:
            print(f"  {lo}..{hi} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} n={s['n_dates']}")

print("\n--- turnover ---")
for name in ['mom20_skip3_div_rv20', 'mom60_skip5_div_rv60']:
    f = cands[name].loc['2021-01-01':]
    print(f"{name:32s} signal_turnover={turnover_signal(f):.4f}")
