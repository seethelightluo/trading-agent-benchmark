"""miner_1 cycle 2026-07-30 (v12): final candidate batch.

Fixes and new combinations:
 1) down_corr_asym_60 (mp=10): conditional correlation asymmetry - retry with looser min_periods
 2) down_beta_asym_60 (mp=10): conditional BETA asymmetry (down-market beta - up-market beta)
 3) vol_ratio_20x60 (mp=8/20): vol term-structure slope (retry with union-calendar-aware mp)
 4) overnight_minus_intraday_20: gap persistence vs intraday fade differential
 5) kurt_20: short-window tail fatness (m4/m2^2 - 3 over 20d)

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

cands = {}

# 1) & 2) conditional corr/beta asymmetry, vectorized, mp=10
ew_ret = ret.mean(axis=1)
def cond_moments(asset_ret, mkt_ret, cond, win=60, mp=10):
    I = cond.astype(float).replace(0.0, np.nan)
    x, y = asset_ret, mkt_ret
    E1 = I.rolling(win, min_periods=mp).mean()
    Ex = (x * I).rolling(win, min_periods=mp).mean()
    Ey = (y * I).rolling(win, min_periods=mp).mean()
    Exx = (x * x * I).rolling(win, min_periods=mp).mean()
    Eyy = (y * y * I).rolling(win, min_periods=mp).mean()
    Exy = (x * y * I).rolling(win, min_periods=mp).mean()
    return E1, Ex, Ey, Exx, Eyy, Exy

def build_asym():
    cdn = {}
    cup = {}
    bdn = {}
    bup = {}
    for a in ret.columns:
        E1d, Exd, Eyd, Exxd, Eyyd, Exyd = cond_moments(ret[a], ew_ret, (ew_ret < 0))
        E1u, Exu, Eyu, Exxu, Eyyu, Exyu = cond_moments(ret[a], ew_ret, (ew_ret > 0))
        covd = Exyd / E1d - (Exd / E1d) * (Eyd / E1d)
        vxd = Exxd / E1d - (Exd / E1d) ** 2
        vyd = Eyyd / E1d - (Eyd / E1d) ** 2
        covu = Exyu / E1u - (Exu / E1u) * (Eyu / E1u)
        vxu = Exxu / E1u - (Exu / E1u) ** 2
        vyu = Eyyu / E1u - (Eyu / E1u) ** 2
        cdn[a] = covd / np.sqrt(vxd * vyd)
        cup[a] = covu / np.sqrt(vxu * vyu)
        bdn[a] = covd / vyd
        bup[a] = covu / vyu
    return (pd.DataFrame(cdn).reindex(ret.index), pd.DataFrame(cup).reindex(ret.index),
            pd.DataFrame(bdn).reindex(ret.index), pd.DataFrame(bup).reindex(ret.index))

cdn, cup, bdn, bup = build_asym()
cands['down_corr_asym_60'] = cdn - cup
cands['down_beta_asym_60'] = bdn - bup

# 3) vol_ratio_20x60
vol20 = ret.rolling(20, min_periods=8).std()
vol60 = ret.rolling(60, min_periods=20).std()
cands['vol_ratio_20x60'] = vol20 / vol60 - 1.0

# 4) overnight_minus_intraday_20
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
overnight = opn / close.shift(1) - 1.0
cands['overnight_minus_intraday_20'] = (overnight - intraday).rolling(20, min_periods=8).mean()

# 5) kurt_20
mu20 = ret.rolling(20, min_periods=8).mean()
m2_20 = ((ret - mu20) ** 2).rolling(20, min_periods=8).mean()
m4_20 = ((ret - mu20) ** 4).rolling(20, min_periods=8).mean()
cands['kurt_20'] = m4_20 / (m2_20 ** 2) - 3.0

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

with open('scripts/miner_1_20260730_explore_v12_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE saved scripts/miner_1_20260730_explore_v12_results.json")
