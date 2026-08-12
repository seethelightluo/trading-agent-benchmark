"""
miner1 2028-06-09: explore combined trend+vol conditioned momentum (third round).
Hypothesis: momentum whipsaws in risk-off because extended high-vol names keep
ranking top. Conditioning momentum on trend state AND scaling by realized vol
should jointly de-risk top picks. Baseline mom_120d_skip5 included for comparison.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner1_ic_lib import load_panel, ic_series, fwd_returns, summarize_ic, coverage_stats, turnover_signal

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
m120 = mom_skip(close, 120, 5)
m60 = mom_skip(close, 60, 5)
rv60 = rv(close, 60)
rv20 = rv(close, 20)
sma20, sma60, sma100 = sma(close, 20), sma(close, 60), sma(close, 100)
donch120 = close.rolling(120).max().shift(1)

# trend-conditioned AND vol-scaled
cands['mom120_x_tr60_div_rv60'] = m120 * (close > sma60).astype(float) / rv60
cands['mom120_x_tr100_div_rv60'] = m120 * (close > sma100).astype(float) / rv60
cands['mom120_x_tr60x20_div_rv60'] = m120 * ((close > sma60) & (close > sma20)).astype(float) / rv60
# trend strength version (clipped)
tr_str = (close / sma60 - 1.0).clip(-0.1, 0.1)
cands['mom120_x_trstr60_div_rv60'] = m120 * tr_str / rv60
# donchian trend + vol scaling
cands['mom120_x_don120_div_rv60'] = m120 * (close > donch120).astype(float) / rv60
# drawdown penalty (avoid extended names): mom * (1 + dd60), dd60 in (-1, 0]
dd60 = close / close.rolling(60).max() - 1.0
cands['mom120_x_dd60pen'] = m120 * (1.0 + dd60)
cands['mom120_x_dd60pen_div_rv60'] = m120 * (1.0 + dd60) / rv60
# vol-scaled momentum with shorter horizon for reference
cands['mom60_x_tr100_div_rv60'] = m60 * (close > sma100).astype(float) / rv60
cands['mom20_skip3_x_tr60_div_rv20'] = mom_skip(close, 20, 3) * (close > sma60).astype(float) / rv20
# baseline
cands['mom_120d_skip5_BASELINE'] = m120

print(f"{'variant':34s} {'IC':>7s} {'ICIR':>7s} {'hit':>6s} {'t':>6s} {'n':>5s} {'covD':>5s} {'avgV':>5s}")
for name, f in cands.items():
    f = f.loc['2021-01-01':]
    ic = ic_series(f, fw[1])
    s = summarize_ic(ic, name)
    cov = coverage_stats(f)
    if s:
        print(f"{name:34s} {s['mean_ic']:+7.4f} {s['icir']:+7.4f} {s['hit_rate']:6.3f} {s['t_stat']:+6.2f} {s['n_dates']:5d} {cov['dates_valid_ge8']:5d} {cov['avg_valid']:5.1f}")

print("\n--- sub-period robustness (1d IC) for promising variants ---")
promising = ['mom120_x_tr60_div_rv60', 'mom120_x_tr100_div_rv60', 'mom120_x_tr60x20_div_rv60',
             'mom120_x_don120_div_rv60', 'mom120_x_dd60pen', 'mom120_x_dd60pen_div_rv60', 'mom_120d_skip5_BASELINE']
for name in promising:
    f = cands[name]
    print(name)
    for lo, hi in [('2021-01-01', '2022-12-31'), ('2023-01-01', '2024-12-31'),
                   ('2025-01-01', '2026-07-15'), ('2026-07-16', '2027-12-31'),
                   ('2028-01-01', '2028-06-08')]:
        ff = f.loc[lo:hi]
        ic = ic_series(ff, fw[1].loc[lo:hi])
        s = summarize_ic(ic, f'{lo}..{hi}')
        if s:
            print(f"  {lo}..{hi} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} n={s['n_dates']}")

print("\n--- decay (IC by horizon) for promising variants ---")
for name in promising:
    f = cands[name].loc['2021-01-01':]
    print(name)
    for h in [1, 2, 3, 5, 10]:
        ic = ic_series(f, fw[h])
        s = summarize_ic(ic, f'h{h}')
        if s:
            print(f"  h={h:2d} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} n={s['n_dates']}")

print("\n--- turnover ---")
for name in promising:
    f = cands[name].loc['2021-01-01':]
    print(f"{name:34s} signal_turnover={turnover_signal(f):.4f}")
