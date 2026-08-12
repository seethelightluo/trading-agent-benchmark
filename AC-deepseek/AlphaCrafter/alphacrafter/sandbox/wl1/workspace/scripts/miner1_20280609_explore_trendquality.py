"""
miner1 2028-06-09: explore trend-quality / momentum-carry candidates (round 4).
Goal: find a factor with stable full-sample 1d |ICIR| >= 0.084 (not just recent-regime).
Families: Kaufman efficiency ratio, momentum carry (12-1, slope), trend persistence,
proximity-to-high (drawdown). Also diagnose why vol-division kills coverage.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner1_ic_lib import load_panel, ic_series, fwd_returns, summarize_ic, coverage_stats, turnover_signal

panel = load_panel()
close = panel['close']
fw = fwd_returns(close, horizons=(1, 2, 3, 5, 10))


def mom_skip(px, n, skip):
    return px.shift(skip) / px.shift(skip + n) - 1.0


def sma(px, n):
    return px.rolling(n).mean()


# --- diagnostic: why does division by rv kill coverage? ---
rv60 = close.pct_change().rolling(60).std()
m120 = mom_skip(close, 120, 5)
print("DIAG valid counts (>=2021):")
print("  m120:            ", int(m120.loc['2021-01-01':].notna().sum(axis=1).ge(8).sum()))
print("  rv60:            ", int(rv60.loc['2021-01-01':].notna().sum(axis=1).ge(8).sum()))
q = (m120 / rv60).loc['2021-01-01':]
print("  m120/rv60:       ", int(q.notna().sum(axis=1).ge(8).sum()), " avg_valid=", round(q.notna().sum(axis=1).mean(), 2))
print("  rv60 zero count: ", int((rv60 == 0).sum().sum()), " rv60 NaN count:", int(rv60.isna().sum().sum()))
print("  m120 zero&rv0 0/0 -> NaN count:", int(((rv60 == 0) & (m120 == 0)).sum().sum()))

# --- candidates ---
cands = {}
# Kaufman efficiency ratio: |close - close_{t-n}| / sum of |daily moves| over n
def eff_ratio(px, n):
    num = (px - px.shift(n)).abs()
    den = px.diff().abs().rolling(n).sum()
    return num / den

cands['er20'] = eff_ratio(close, 20)
cands['er60'] = eff_ratio(close, 60)
cands['er120'] = eff_ratio(close, 120)
# momentum carry 12-1 and slope
cands['mom121_carry'] = mom_skip(close, 120, 5) - mom_skip(close, 20, 3)
cands['mom_slope_120_60'] = mom_skip(close, 120, 5) - mom_skip(close, 60, 5)
# trend persistence: fraction of last 60d where close > SMA20
cands['trend_pers60'] = (close > sma(close, 20)).rolling(60).mean()
# proximity to 120d high
cands['dd120'] = close / close.rolling(120).max() - 1.0
cands['dd252'] = close / close.rolling(252).max() - 1.0
# baseline for reference
cands['mom_120d_skip5_BASELINE'] = m120

print(f"\n{'variant':28s} {'IC':>7s} {'ICIR':>7s} {'hit':>6s} {'t':>6s} {'n':>5s} {'covD':>5s} {'avgV':>5s}")
for name, f in cands.items():
    f = f.loc['2021-01-01':]
    ic = ic_series(f, fw[1])
    s = summarize_ic(ic, name)
    cov = coverage_stats(f)
    if s:
        print(f"{name:28s} {s['mean_ic']:+7.4f} {s['icir']:+7.4f} {s['hit_rate']:6.3f} {s['t_stat']:+6.2f} {s['n_dates']:5d} {cov['dates_valid_ge8']:5d} {cov['avg_valid']:5.1f}")

print("\n--- sub-period robustness (1d IC) ---")
for name in ['er60', 'er120', 'mom121_carry', 'mom_slope_120_60', 'trend_pers60', 'dd120', 'dd252']:
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

print("\n--- decay (IC by horizon) for best candidates ---")
for name in ['er60', 'er120', 'mom121_carry', 'mom_slope_120_60', 'trend_pers60', 'dd120']:
    f = cands[name].loc['2021-01-01':]
    print(name)
    for h in [1, 2, 3, 5, 10]:
        ic = ic_series(f, fw[h])
        s = summarize_ic(ic, f'h{h}')
        if s:
            print(f"  h={h:2d} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} n={s['n_dates']}")

print("\n--- turnover ---")
for name in ['er60', 'er120', 'mom121_carry', 'mom_slope_120_60', 'trend_pers60', 'dd120']:
    f = cands[name].loc['2021-01-01':]
    print(f"{name:28s} signal_turnover={turnover_signal(f):.4f}")
