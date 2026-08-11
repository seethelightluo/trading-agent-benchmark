"""miner_1 cycle 2026-07-30 (v11): re-run candidates with union-calendar-aware min_periods.

Root cause found in v10: closes_panel is a union of 5+ trading calendars -> each non-crypto
asset has ~45% NaN rows (its own holidays). rolling(60, min_periods=36) on RETURNS therefore
produced valid values on only ~17-228 dates per asset (BTC/ETH nearly complete), which is why
max_ret_60 / skew_60 / kurt_60 / down_corr_asym_60 had tiny coverage or 0 IC dates.

Fix: min_periods scaled to ~1/3 of window (20 for 60d, 8 for 20d). Candidates:

 1) max_ret_60       : 60d max daily return (lottery/MAX), min_periods=20
 2) max_ret_vol_60   : max_ret_60 / vol_60 (vol-normalized lottery)
 3) skew_60          : rolling 60d skewness m3/m2^1.5
 4) kurt_60          : rolling 60d excess kurtosis m4/m2^2 - 3
 5) down_corr_asym_60: corr(asset, EW mkt | mkt down) - corr(asset, EW | mkt up), 60d,
                        fully vectorized via rolling moments of products
 6) range_vol_ratio_20: avg (high-low)/close over 20d / 20d close-to-close vol (retry)
 7) avg_intraday_10 / avg_intraday_60 / avg_overnight_20: session-drift variants
 8) range_pos_10     : 10d range position (short-window recovery phase)

Gate: |IC|>=0.007, |ICIR|>=0.084, library max-abs Spearman rho < 0.5 (artifact-based).
"""
import json, sys, time, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import closes_panel, forward_returns, ic_series, summary_metrics, regime_split
from miner3_lib import decode_artifact, LIB_FACTORS

VIS = '2026-07-29'
H = 10
close = closes_panel(VIS)
ret = close.pct_change()
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS}", flush=True)

# ---------- raw ingredients ----------
vol60 = ret.rolling(60, min_periods=20).std()
vol20 = ret.rolling(20, min_periods=8).std()
mu60 = ret.rolling(60, min_periods=20).mean()

cands = {}

# 1) max_ret_60
cands['max_ret_60'] = ret.rolling(60, min_periods=20).max()

# 2) max_ret_vol_60
cands['max_ret_vol_60'] = cands['max_ret_60'] / vol60

# 3) skew_60
m2 = ((ret - mu60) ** 2).rolling(60, min_periods=20).mean()
m3 = ((ret - mu60) ** 3).rolling(60, min_periods=20).mean()
cands['skew_60'] = m3 / (m2 ** 1.5)

# 4) kurt_60
m4 = ((ret - mu60) ** 4).rolling(60, min_periods=20).mean()
cands['kurt_60'] = m4 / (m2 ** 2) - 3.0

# 5) down_corr_asym_60 (vectorized conditional correlation)
ew_ret = ret.mean(axis=1)
def cond_corr(asset_ret, mkt_ret, cond, win=60, mp=20):
    I = (cond).astype(float).replace(0.0, np.nan)
    x = asset_ret; y = mkt_ret
    E1 = I.rolling(win, min_periods=mp).mean()
    Ex = (x * I).rolling(win, min_periods=mp).mean()
    Ey = (y * I).rolling(win, min_periods=mp).mean()
    Exx = (x * x * I).rolling(win, min_periods=mp).mean()
    Eyy = (y * y * I).rolling(win, min_periods=mp).mean()
    Exy = (x * y * I).rolling(win, min_periods=mp).mean()
    cov = Exy / E1 - (Ex / E1) * (Ey / E1)
    vx = Exx / E1 - (Ex / E1) ** 2
    vy = Eyy / E1 - (Ey / E1) ** 2
    return cov / np.sqrt(vx * vy)
I_dn = (ew_ret < 0)
I_up = (ew_ret > 0)
down_corr, up_corr = {}, {}
for a in ret.columns:
    down_corr[a] = cond_corr(ret[a], ew_ret, I_dn)
    up_corr[a] = cond_corr(ret[a], ew_ret, I_up)
down_corr = pd.DataFrame(down_corr).reindex(ret.index)
up_corr = pd.DataFrame(up_corr).reindex(ret.index)
cands['down_corr_asym_60'] = down_corr - up_corr

# 6) range_vol_ratio_20 (retry with min_periods=8)
import os as _os
def ohlc_panel(symbols, vis):
    out = {}
    for s in symbols:
        fp = _os.path.join('../persistent/stock_data', s + '.csv')
        if not _os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=['date'])
        df = df[df['date'] <= pd.Timestamp(vis)].set_index('date')
        out[s] = df[['open', 'high', 'low', 'close']].astype(float)
    return pd.concat(out, axis=1)
ohlc = ohlc_panel(close.columns, VIS)
hi = ohlc.xs('high', axis=1, level=1)
lo = ohlc.xs('low', axis=1, level=1)
rng = (hi - lo) / close
cands['range_vol_ratio_20'] = rng.rolling(20, min_periods=8).mean() / ret.rolling(20, min_periods=8).std()

# 7) session drift variants
opn = ohlc.xs('open', axis=1, level=1)
intraday = close / opn - 1.0
overnight = opn / close.shift(1) - 1.0
cands['avg_intraday_10'] = intraday.rolling(10, min_periods=6).mean()
cands['avg_intraday_60'] = intraday.rolling(60, min_periods=20).mean()
cands['avg_overnight_20'] = overnight.rolling(20, min_periods=8).mean()

# 8) range_pos_10
cands['range_pos_10'] = (close - close.rolling(10, min_periods=6).min()) / \
                        (close.rolling(10, min_periods=6).max() - close.rolling(10, min_periods=6).min())

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

with open('scripts/miner_1_20260730_explore_v11_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE saved scripts/miner_1_20260730_explore_v11_results.json")
