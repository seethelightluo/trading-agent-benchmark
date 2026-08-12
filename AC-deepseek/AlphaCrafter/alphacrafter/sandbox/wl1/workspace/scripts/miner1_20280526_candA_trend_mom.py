"""miner1 2028-05-26: candidate A - trend-conditioned momentum (mom_trend_cond).
Hypothesis: momentum whipsaws in risk-off; conditioning momentum on long-term
trend state (close vs SMA) should improve predictive power and stability.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner1_ic_lib import load_panel, ic_series, fwd_returns, summarize_ic, coverage_stats, turnover_rank, turnover_signal, WATCH

panel = load_panel()
close = panel['close']
ret = panel['ret']

# forward returns (1d used for admission)
fw = fwd_returns(close, horizons=(1, 2, 3, 5, 10))

def mom_skip(px, n, skip):
    return px.shift(skip) / px.shift(skip + n) - 1.0

def sma(px, n):
    return px.rolling(n).mean()

cands = {}
# V1: 120d momentum x I(close > SMA60)
m120 = mom_skip(close, 120, 5)
cands['mom120_skip5_x_trend60'] = m120 * (close > sma(close, 60)).astype(float)
# V2: 120d momentum x trend-strength multiplier (close/SMA60 - 1)
cands['mom120_skip5_x_trend60_str'] = m120 * (close / sma(close, 60) - 1.0)
# V3: 120d momentum x I(close > SMA100)
cands['mom120_skip5_x_trend100'] = m120 * (close > sma(close, 100)).astype(float)
# V4: 60d momentum x I(close > SMA120)
m60 = mom_skip(close, 60, 5)
cands['mom60_skip5_x_trend120'] = m60 * (close > sma(close, 120)).astype(float)
# V5: 120d momentum x I(close > SMA120)
cands['mom120_skip5_x_trend120'] = m120 * (close > sma(close, 120)).astype(float)
# V6: 120d momentum x I(close > SMA200)
cands['mom120_skip5_x_trend200'] = m120 * (close > sma(close, 200)).astype(float)

for name, f in cands.items():
    f = f.loc['2021-01-01':]
    ic = ic_series(f, fw[1])
    s = summarize_ic(ic, name)
    cov = coverage_stats(f)
    if s:
        print(f"{name:34s} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} t={s['t_stat']:+.2f} n={s['n_dates']} cov_dates={cov['dates_valid_ge8']} avg_valid={cov['avg_valid']:.1f}")

print("\n--- decay for best variants (IC by horizon) ---")
for name in ['mom120_skip5_x_trend60', 'mom120_skip5_x_trend100']:
    f = cands[name].loc['2021-01-01':]
    print(name)
    for h in [1, 2, 3, 5, 10]:
        ic = ic_series(f, fw[h])
        s = summarize_ic(ic, f'h{h}')
        if s:
            print(f"  h={h:2d} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} n={s['n_dates']}")

print("\n--- sub-period robustness (1d IC) ---")
for name in ['mom120_skip5_x_trend60', 'mom120_skip5_x_trend100']:
    f = cands[name]
    print(name)
    for lo, hi in [('2021-01-01', '2022-12-31'), ('2023-01-01', '2024-12-31'), ('2025-01-01', '2026-07-15'), ('2026-07-16', '2027-12-31'), ('2028-01-01', '2028-05-25')]:
        ff = f.loc[lo:hi]
        ic = ic_series(ff, fw[1].loc[lo:hi])
        s = summarize_ic(ic, f'{lo}..{hi}')
        if s:
            print(f"  {lo}..{hi} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} n={s['n_dates']}")

print("\n--- turnover ---")
for name in ['mom120_skip5_x_trend60', 'mom120_skip5_x_trend100']:
    f = cands[name].loc['2021-01-01':]
    print(f"{name:34s} rank_turnover={turnover_rank(f):.4f} signal_turnover={turnover_signal(f):.4f}")
