"""miner_3 2026-08-11: explore trend-efficiency & market/bond-sensitivity family (v2, per-asset calendars).

Candidates (15-asset tradable cross-asset universe, admission h=10, min_valid=8):
  - ER_20        : Kaufman efficiency ratio, 20d (|net move| / path length)
  - ER_60        : Kaufman efficiency ratio, 60d
  - MKTBETA_60   : rolling 60d beta of each asset to equal-weight universe return
  - RESID_MOM_20 : 20d cumulative return of market-beta-residualized daily returns
  - BOND_BETA_60 : rolling 60d beta of each asset to US10Y daily return (duration sensitivity)
"""
import sys, json, os
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split)
sys.path.insert(0, '/home/lxx/trade-agent-benchmark/AC-deepseek/AlphaCrafter')
from alphacrafter import factor_contract as fc

VIS = '2026-08-10'
H = 10
close = closes_panel(VIS)
ret = close.pct_change()
fr = forward_returns(close, H)
market = ret.mean(axis=1)
print(f'visible_through={VIS}  assets={close.shape[1]}  rows={close.shape[0]}')


def kaufman_er(n, minp_frac=0.6):
    out = {}
    for a in close.columns:
        s = close[a].dropna()
        r = s.pct_change()
        net = (s / s.shift(n) - 1.0).abs()
        path = r.abs().rolling(n, min_periods=max(10, int(n * minp_frac))).sum()
        er = (net / path).clip(upper=1.0)
        out[a] = er
    return pd.DataFrame(out).reindex(close.index)


def rolling_beta(a_ret, m_ret, win, minp=40):
    out = {}
    for a in a_ret.columns:
        pair = pd.concat([a_ret[a].rename('a'), m_ret.rename('m')], axis=1).dropna()
        b = pair['a'].rolling(win, min_periods=minp).cov(pair['m']) / pair['m'].rolling(win, min_periods=minp).var()
        out[a] = b
    return pd.DataFrame(out).reindex(a_ret.index)


signals = {}
signals['ER_20'] = kaufman_er(20)
signals['ER_60'] = kaufman_er(60)
signals['MKTBETA_60'] = rolling_beta(ret, market, 60)
resid = pd.DataFrame({a: ret[a] - signals['MKTBETA_60'][a] * market for a in ret.columns}).reindex(ret.index)
resid_mom = {}
for a in resid.columns:
    s = resid[a].dropna()
    resid_mom[a] = s.rolling(20, min_periods=10).sum()
signals['RESID_MOM_20'] = pd.DataFrame(resid_mom).reindex(resid.index)
us10y = close['US10Y']
signals['BOND_BETA_60'] = rolling_beta(ret, us10y.pct_change(), 60)

print('\n=== VALIDATION (h=%d, min_valid=8) ===' % H)
results = {}
for fid, sig in signals.items():
    ics = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ics, sig, fr, close, h=H)
    reg = regime_split(ics)
    ic, icir = m['ic'], m['icir']
    passed = abs(ic) >= 0.007 and abs(icir) >= 0.084
    results[fid] = {'ic': ic, 'icir': icir, 'n': m['n_ic_dates'], 'regime': reg, 'pass': bool(passed)}
    print(f"{fid:16s} IC={ic:+.4f} ICIR={icir:+.4f} n={m['n_ic_dates']:4d} "
          f"hit={m['ic_hit_ratio']:.3f} cov={m['coverage_asset_days']:.3f} "
          f"cov_dates={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.3f} PASS={passed}")
    print(f"    decay: {m['decay_ic_by_horizon']}")
    print(f"    regime: 2020-22 IC={reg['2020-2022']['ic']:+.4f} (ICIR {reg['2020-2022']['icir']:+.3f}) | "
          f"2023-24 IC={reg['2023-2024']['ic']:+.4f} (ICIR {reg['2023-2024']['icir']:+.3f}) | "
          f"2025-26 IC={reg['2025-2026']['ic']:+.4f} (ICIR {reg['2025-2026']['icir']:+.3f})")

print('\n=== REDUNDANCY vs LIBRARY (max abs rho per candidate) ===')
lib_files = [f for f in sorted(os.listdir('factors')) if f.endswith('.json') and f != 'factor_ensemble.json']
rho_map = {}
for fid, sig in signals.items():
    arr = sig.values.astype(float)
    maxrho, withf = 0.0, None
    for lf in lib_files:
        try:
            payload = json.load(open(os.path.join('factors', lf)))
            sarr = fc._load_signal_artifact(payload, os.path.join('factors', lf))
            if sarr is None:
                continue
            rho = fc._pairwise_abs_spearman(arr, sarr)
            if rho > maxrho:
                maxrho, withf = rho, lf
        except Exception:
            continue
    rho_map[fid] = round(maxrho, 3)
    print(f"{fid:16s} max_rho={maxrho:.3f} vs {withf}")

with open('scripts/miner3_20260811_explore_eff_mktbeta_results.json', 'w') as f:
    json.dump({'visible_through': VIS, 'results': results, 'library_max_rho': rho_map}, f, indent=1)
print('\nsaved results json')
