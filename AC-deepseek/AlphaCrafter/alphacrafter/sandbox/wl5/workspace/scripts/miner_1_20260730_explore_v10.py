"""miner_1 cycle 2026-07-30 (v10): post-dd_pos_60-eviction candidate exploration.

Context: dd_pos_60 passed IC/ICIR (0.0323/0.0964) but was evicted for redundancy with
semi_down_ratio_20 (abs Spearman rho ~0.65). v10 tests structurally-distinct candidates and
an orthogonalized rescue:

 1) dd_orth_60  : per-date cross-sectional orthogonalization of dd_pos_60 against
                  semi_down_ratio_20 (residual of range-position after removing
                  downside-ratio component) -> should keep IC, kill the redundancy.
 2) max_ret_vol_60: 60d max daily return / 60d vol (vol-normalized lottery / MAX effect);
                  also debug the suspiciously low coverage of raw max_ret_60.
 3) avg_intraday_20: mean((close-open)/open, 20d) - intraday session drift.
 4) skew_60     : rolling 60d skewness of daily returns (m3/m2^1.5).

All research restricted to visible window <= 2026-07-29. Gate: |IC|>=0.007, |ICIR|>=0.084,
library max-abs Spearman rho < 0.5 (computed from real signal artifacts).
"""
import json, sys, time, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import closes_panel, forward_returns, ic_series, summary_metrics, regime_split
from miner3_lib import decode_artifact, LIB_FACTORS

VIS = '2026-07-29'
H = 10
t0 = time.time()
close = closes_panel(VIS)
ret = close.pct_change()
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS} load={time.time()-t0:.1f}s", flush=True)

# debug max_ret_60 coverage
mr = ret.rolling(60, min_periods=36).max()
print("max_ret_60 per-asset valid counts:", dict(mr.notna().sum()), flush=True)

# ---- candidate signals ----
# 1) dd_orth_60: per-date orth of range position on semi_down_ratio_20 (artifact)
semi = decode_artifact(json.load(open('factors/semi_down_ratio_20.json'))['validation']['signal_artifact'])
semi = semi.reindex(close.index)
rmax60 = close.rolling(60, min_periods=36).max()
rmin60 = close.rolling(60, min_periods=36).min()
dd_pos = (close - rmin60) / (rmax60 - rmin60)
dd_orth = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
for d in close.index:
    x = dd_pos.loc[d]
    y = semi.loc[d]
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 8:
        continue
    rx = x[m].rank()
    ry = y[m].rank()
    zx = (rx - rx.mean()) / rx.std(ddof=0)
    zy = (ry - ry.mean()) / ry.std(ddof=0)
    beta = float(zx.corr(zy))
    if not np.isfinite(beta):
        continue
    resid = zx - beta * zy
    dd_orth.loc[d, m[m].index] = resid.values

# 2) max_ret_vol_60
vol60 = ret.rolling(60, min_periods=36).std()
max_ret_vol_60 = mr / vol60

# 3) avg_intraday_20 (uses open panel)
import os as _os
def open_panel(symbols, vis):
    out = {}
    for s in symbols:
        fp = _os.path.join('../persistent/stock_data', s + '.csv')
        if not _os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=['date'])
        df = df[df['date'] <= pd.Timestamp(vis)].set_index('date')
        out[s] = df['open'].astype(float)
    return pd.DataFrame(out).sort_index()
opn = open_panel(close.columns, VIS)
intraday = close / opn - 1.0
avg_intraday_20 = intraday.rolling(20, min_periods=12).mean()

# 4) skew_60
mu = ret.rolling(60, min_periods=36).mean()
m2 = ((ret - mu) ** 2).rolling(60, min_periods=36).mean()
m3 = ((ret - mu) ** 3).rolling(60, min_periods=36).mean()
skew_60 = m3 / (m2 ** 1.5)

cands = {
    'dd_orth_60': dd_orth,
    'max_ret_vol_60': max_ret_vol_60,
    'avg_intraday_20': avg_intraday_20,
    'skew_60': skew_60,
}

fr = forward_returns(close, H)
results = {}
for fid, sig in cands.items():
    t0 = time.time()
    ic_s = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic_s, sig, fr, close, h=H)
    if m is None:
        print(f"{fid}: INSUFFICIENT ({len(ic_s)} ic dates) [{time.time()-t0:.1f}s]", flush=True)
        results[fid] = {"gate_pass": False, "reason": "insufficient", "n_ic_dates": len(ic_s)}
        continue
    m['regime'] = regime_split(ic_s)
    # library max-abs Spearman rho from real artifacts
    best = 0.0
    rhos = {}
    for lfid in LIB_FACTORS:
        p = f'factors/{lfid}.json'
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        art = d.get('validation', {}).get('signal_artifact')
        if not art:
            continue
        libp = decode_artifact(art).reindex(close.index)
        common = sig.index.intersection(libp.index)
        a = sig.loc[common].stack()
        b = libp.loc[common].stack()
        mm = a.notna() & b.notna()
        if mm.sum() >= 200:
            r = float(a[mm].rank().corr(b[mm].rank()))
            if np.isfinite(r):
                rhos[lfid] = round(r, 3)
                best = max(best, abs(r))
    m['library_spearman_rho'] = rhos
    m['max_abs_library_correlation'] = round(best, 3)
    gate = abs(m['ic']) >= 0.007 and abs(m['icir'] or 0) >= 0.084 and best < 0.5
    m['gate_pass'] = bool(gate)
    results[fid] = m
    print(f"=== {fid}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
          f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} max_rho_lib={m['max_abs_library_correlation']} GATE={gate} [{time.time()-t0:.1f}s]", flush=True)
    print("  decay:", m['decay_ic_by_horizon'], flush=True)
    print("  regimes:", m['regime'], flush=True)
    if gate:
        sig.index = sig.index.strftime('%Y-%m-%d')
        os.makedirs('scripts/_panels', exist_ok=True)
        sig.to_csv(f'scripts/_panels/{fid}.csv')

with open('scripts/miner_1_20260730_explore_v10_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE saved scripts/miner_1_20260730_explore_v10_results.json")
