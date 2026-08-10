"""Round 8: novel factor candidates vs full 12-factor library artifacts.

Ideas (single-idea constructs, each validated separately):
 1. obv_slope_20       - volume-weighted trend (On-Balance-Volume linear slope)
 2. spx_corr_60        - rolling return correlation with SPX (diversification)
 3. spill_btc_20x10    - BTC-beta x BTC momentum (crypto spillover)
 4. weekday_eff_52     - day-of-week seasonal consistency (52w rolling)
 5. month_eff_36       - calendar-month seasonal consistency (36m rolling)
 6. capture_eff_20     - 20d return per unit of cumulative daily range
 7. rsi_60             - long-horizon RSI(60)
 8. rel_strength_spx_20- asset 20d momentum minus SPX 20d momentum
 9. vol_conc_20        - volume Herfindahl concentration over 20d
10. mom_cond_vix_low   - momentum gated by low-VIX regime (macro-conditional)
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor)

np.seterr(all='ignore')

# ---------- load data ----------
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices loaded: {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})")
vix = load_index('VIX', prices=prices)
print(f"VIX loaded: {vix is not None}, rows={0 if vix is None else len(vix)}")

# ---------- library artifacts (real signal matrices) ----------
lib = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            art = d.get('signal_artifact')
            if art and Path('factors', art).exists():
                lib[d['factor_id']] = np.load(Path('factors', art))
    except Exception as e:
        print("lib skip", f, e)
print(f"library artifacts loaded: {len(lib)} -> {sorted(lib)}")

def max_lib_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, la in lib.items():
        corrs = []
        for i in range(arr.shape[0]):
            x, y = arr[i], la[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                r = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
                if np.isfinite(r):
                    corrs.append(r)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

def cand_corr(panels):
    """mean daily cross-sectional Spearman corr between candidates (top-left)."""
    ids = list(panels.keys())
    M = np.full((len(ids), len(ids)), np.nan)
    arrs = {k: signal_matrix(v, grid) for k, v in panels.items()}
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            corrs = []
            for t in range(arrs[ids[i]].shape[0]):
                x, y = arrs[ids[i]][t], arrs[ids[j]][t]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() >= 8:
                    r = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
                    if np.isfinite(r):
                        corrs.append(r)
            if corrs:
                M[i, j] = M[j, i] = float(np.mean(corrs))
    return ids, M

# ---------- candidate definitions ----------
def obv_slope_20(df, s):
    ret = df['close'].pct_change()
    vol = df['volume'].replace(0, np.nan)
    obv = (np.sign(ret) * vol).fillna(0.0).cumsum()
    def slope(x):
        if len(x) < 20 or np.isnan(x).any():
            return np.nan
        yy = x.values.astype(float)
        d = np.diff(yy)
        sd = np.std(d)
        if not np.isfinite(sd) or sd <= 0:
            return np.nan
        t = np.arange(len(yy))
        b = np.polyfit(t, yy, 1)[0]
        return b / sd
    return obv.rolling(20).apply(slope, raw=False)

def spx_corr_60(df, s):
    if s == 'SPX':
        return pd.Series(np.nan, index=df.index)
    spx = prices['SPX']['close'].reindex(df.index)
    r = df['close'].pct_change()
    rs = spx.pct_change()
    z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
    return z['r'].rolling(60).corr(z['s'])

def spill_btc_20x10(df, s):
    if s == 'BTC':
        return pd.Series(np.nan, index=df.index)
    btc = prices['BTC']['close'].reindex(df.index)
    r = df['close'].pct_change()
    rb = btc.pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1).dropna()
    beta = z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var()
    momb = btc.shift(2) / btc.shift(12) - 1.0
    return (beta * momb).reindex(z.index)

def weekday_eff_52(df, s):
    ret = df['close'].pct_change()
    wd = df.index.dayofweek
    out = {}
    for i in range(len(df)):
        d = df.index[i]
        m = (wd == d.dayofweek) & (df.index >= d - pd.Timedelta(days=364)) & (df.index < d)
        vals = ret[m]
        out[d] = vals.mean() if len(vals) >= 10 else np.nan
    return pd.Series(out)

def month_eff_36(df, s):
    ret = df['close'].pct_change()
    mo = df.index.month
    out = {}
    for i in range(len(df)):
        d = df.index[i]
        m = (mo == d.month) & (df.index >= d - pd.Timedelta(days=1095)) & (df.index < d)
        vals = ret[m]
        out[d] = vals.mean() if len(vals) >= 12 else np.nan
    return pd.Series(out)

def capture_eff_20(df, s):
    ret = df['close'].pct_change()
    rng = (df['high'] - df['low']) / df['close']
    cum = ret.rolling(20).sum()
    cumr = rng.rolling(20).sum()
    return cum / cumr

def rsi_60(df, s):
    delta = df['close'].diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    au = up.rolling(60).mean()
    ad = dn.rolling(60).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def rel_strength_spx_20(df, s):
    spx = prices['SPX']['close'].reindex(df.index)
    a = df['close'].shift(5) / df['close'].shift(25) - 1.0
    m = spx.shift(5) / spx.shift(25) - 1.0
    return a - m

def vol_conc_20(df, s):
    vol = df['volume'].replace(0, np.nan)
    def hhi(x):
        x = x[~np.isnan(x)]
        if len(x) < 10 or x.sum() <= 0:
            return np.nan
        sh = x / x.sum()
        return float((sh ** 2).sum())
    return vol.rolling(20).apply(hhi, raw=True)

def mom_cond_vix_low(df, s):
    if vix is None:
        return None
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = vix['close'].reindex(df.index)
    gate = (v < v.rolling(60).median()).astype(float)
    return mom * gate.fillna(0.0)

candidates = {
    'obv_slope_20': obv_slope_20,
    'spx_corr_60': spx_corr_60,
    'spill_btc_20x10': spill_btc_20x10,
    'weekday_eff_52': weekday_eff_52,
    'month_eff_36': month_eff_36,
    'capture_eff_20': capture_eff_20,
    'rsi_60': rsi_60,
    'rel_strength_spx_20': rel_strength_spx_20,
    'vol_conc_20': vol_conc_20,
    'mom_cond_vix_low': mom_cond_vix_low,
}

# ---------- validate ----------
results = {}
panels = {}
for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    panels[fid] = panel
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT DATA (panel {panel.shape})")
        continue
    rho, fid_lib = max_lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid_lib
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = dict(ok=ok, metrics=m)
    print(f"\n=== {fid} === panel {panel.shape} "
          f"range {panel.index.min().date()}..{panel.index.max().date()}")
    print(f"IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"maxlibrho={rho:.3f}({fid_lib})")
    print("decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f} {'PASS' if abs(m['ic'])>=0.007 else 'FAIL'} | "
          f"|ICIR|={abs(m['icir']):.4f} {'PASS' if abs(m['icir'])>=0.084 else 'FAIL'} | "
          f"rho={rho:.3f} {'PASS' if rho<0.5 else 'FAIL'} -> {'PASS' if ok else 'FAIL'}")

print("\n=== candidate-candidate mean daily cross-sectional Spearman rho ===")
ids, M = cand_corr(panels)
hdr = "        " + " ".join(f"{i[:10]:>10}" for i in ids)
print(hdr)
for i, idi in enumerate(ids):
    row = " ".join(f"{M[i,j]:10.2f}" if np.isfinite(M[i,j]) else f"{'-':>10}" for j in range(len(ids)))
    print(f"{idi[:10]:>8} {row}")

json.dump({k: v for k, v in results.items()},
          open('scripts/miner_3_20260730_results_round8.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20260730_results_round8.json")
