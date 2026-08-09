"""miner_3: recompute max_abs_library_correlation for ALL 12 admitted factors
against the FULL current library (12 factors), update persisted factor JSONs,
and rebuild factor_ensemble.json (quality_ic_tilt, top-k<=10).

Context: batch-4 (eff_ratio_60d, amihud_liquidity_20d, ret_autocorr_20d,
dxy_cond_60x20) and batch-5 (btc_spill_cond_60x20, consec_up_ratio_20,
max_ratio_20, usdjpy_beta_cond_60x20) factors were admitted with
max_abs_library_correlation computed against the ORIGINAL 4-factor library only.
With the library now at 12, correlations must be recomputed vs all other 11.

Conventions:
  - primary: miner1/persist_batch4 convention = mean over dates of per-date
    cross-sectional Pearson corr (corrwith axis=1), dates with >=8 valid pairs.
  - reference: per-date cross-sectional Spearman rank corr (screener/batch5 style).
Admission gate: max_abs_library_correlation < 0.5000.
"""
import sys, json, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner1_20260716_lib import build_panel, factor_values, forward_returns, daily_ic, WATCH

H = 10
MIN_VALID = 8
CORR_GATE = 0.5

t0 = time.time()
panel = build_panel()
closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
print(f"panel: grid_dates={len(grid)} assets={len(closes)}  [{time.time()-t0:.0f}s]")

# ---------------- factor builders (mirror persisted library definitions) ----------------
def macro_beta_cond(macro_key, sign=1.0, win=60, mom=20):
    def fn(sym, close, volume, panel=None):
        macro = panel['macro'].get(macro_key)
        if macro is None:
            macro = panel['closes'].get(macro_key)
        if macro is None:
            return None
        g = panel['grid']
        r_a = close.pct_change().reindex(g)
        r_m = macro.pct_change().reindex(g)
        beta = r_a.rolling(win, min_periods=30).cov(r_m) / r_m.rolling(win, min_periods=30).var()
        mm = macro.reindex(g) / macro.shift(mom).reindex(g) - 1.0
        return (sign * beta * mm).replace([np.inf, -np.inf], np.nan)
    return fn

def vix_beta_frame(sym, close, volume):
    macro = panel['macro'].get('VIX')
    g = panel['grid']
    r_a = close.pct_change().reindex(g)
    r_m = macro.pct_change().reindex(g)
    beta = r_a.rolling(60, min_periods=30).cov(r_m) / r_m.rolling(60, min_periods=30).var()
    mm = macro.reindex(g) / macro.shift(20).reindex(g) - 1.0
    return (-1.0 * beta * mm).replace([np.inf, -np.inf], np.nan)

def max_ratio(win=20):
    def fn(sym, close, volume):
        r = close.pct_change().rolling(win)
        return (r.max() / r.min().abs()).replace([np.inf, -np.inf], np.nan)
    return fn

def consec_up_ratio(win=20):
    def fn(sym, close, volume):
        r = (close.pct_change() > 0).astype(float)
        def run_len(x):
            x = x.values
            best_up = best_dn = cur_u = cur_d = 0
            for v in x:
                if v == 1:
                    cur_u += 1; cur_d = 0
                    best_up = max(best_up, cur_u)
                else:
                    cur_d += 1; cur_u = 0
                    best_dn = max(best_dn, cur_d)
            s = best_up + best_dn
            return best_up / s if s > 0 else np.nan
        return r.rolling(win).apply(run_len, raw=False)
    return fn

def autocorr(px, n=20):
    return px.pct_change().rolling(n).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) >= 3 else np.nan, raw=True)

def amihud(win=20):
    def fn(sym, close, volume):
        if volume is None or volume.abs().sum() == 0:
            return None
        illiq = (close.pct_change().abs() / volume.replace(0, np.nan)).rolling(win).mean()
        return (-illiq).replace([np.inf, -np.inf], np.nan)
    return fn

FACTORS = {
    'mom_10d_skip5':       lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
    'mom_120d_skip5':      lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
    'vol_of_vol20x60':     lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
    'vix_beta_cond_60x20': vix_beta_frame,
    'amihud_liquidity_20d': amihud(20),
    'btc_spill_cond_60x20': (lambda f: (lambda s, c, v: f(s, c, v, panel=panel)))(macro_beta_cond('BTC', sign=1.0, win=60, mom=20)),
    'consec_up_ratio_20':  consec_up_ratio(20),
    'dxy_cond_60x20':      (lambda f: (lambda s, c, v: f(s, c, v, panel=panel)))(macro_beta_cond('DXY', sign=1.0, win=60, mom=20)),
    'eff_ratio_60d':       lambda s, c, v: (c - c.shift(60)).abs() / c.diff().abs().rolling(60).sum(),
    'max_ratio_20':        max_ratio(20),
    'ret_autocorr_20d':    lambda s, c, v: -autocorr(c, 20),
    'usdjpy_beta_cond_60x20': (lambda f: (lambda s, c, v: f(s, c, v, panel=panel)))(macro_beta_cond('USDJPY', sign=1.0, win=60, mom=20)),
}

frames = {}
for label, fn in FACTORS.items():
    frames[label] = factor_values(closes, volumes, grid, fn)
    cov = frames[label].notna().mean().mean()
    print(f"[{label}] cov_asset_days={cov:.3f}")
print(f"frames built  [{time.time()-t0:.0f}s]")

# ---------------- per-date cross-sectional correlation (two conventions) ----------------
labels = list(FACTORS.keys())
common_dates = None
for l in labels:
    d = frames[l].dropna(how='all').index
    common_dates = d if common_dates is None else common_dates.intersection(d)
print(f"common dates: {len(common_dates)}")

def pair_corr(fa, fb, method):
    cs = []
    for t in common_dates:
        a, b = fa.loc[t], fb.loc[t]
        mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        if mask.sum() >= MIN_VALID:
            av, bv = a[mask].astype(float), b[mask].astype(float)
            if av.nunique() < 2 or bv.nunique() < 2:
                continue
            if method == 'pearson':
                r = av.corr(bv)
            else:
                r = av.rank().corr(bv.rank())
            if np.isfinite(r):
                cs.append(r)
    return float(np.mean(cs)) if cs else np.nan

corr_pear = pd.DataFrame(index=labels, columns=labels, dtype=float)
corr_spea = pd.DataFrame(index=labels, columns=labels, dtype=float)
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        if i > j:
            continue
        vp = pair_corr(frames[a], frames[b], 'pearson')
        vs = pair_corr(frames[a], frames[b], 'spearman')
        corr_pear.loc[a, b] = corr_pear.loc[b, a] = vp
        corr_spea.loc[a, b] = corr_spea.loc[b, a] = vs
    print(f"  pair loop {i+1}/{len(labels)} [{time.time()-t0:.0f}s]")

pd.set_option('display.width', 250)
print("\n=== PAIRWISE PER-DATE PEARSON CORR (mean over dates) ===")
print(corr_pear.round(3))
print("\n=== PAIRWISE PER-DATE SPEARMAN CORR (mean over dates) ===")
print(corr_spea.round(3))

max_pear = corr_pear.abs().max(axis=1)
max_spea = corr_spea.abs().max(axis=1)
best_pear = corr_pear.abs().idxmax(axis=1)
best_spea = corr_spea.abs().idxmax(axis=1)

print("\n=== MAX ABS LIBRARY CORRELATION (vs other 11 factors) ===")
flags = []
for l in labels:
    print(f"  {l:<24} pearson={max_pear[l]:.4f} (vs {best_pear[l]}) | spearman={max_spea[l]:.4f} (vs {best_spea[l]})")
    if max_pear[l] >= CORR_GATE or max_spea[l] >= CORR_GATE:
        flags.append(l)
print(f"\nFLAGS (>= {CORR_GATE}): {flags if flags else 'none'}")

# ---------------- update persisted factor JSONs ----------------
print("\n=== UPDATING FACTOR JSONS ===")
for l in labels:
    path = f"factors/{l}.json"
    d = json.load(open(path))
    met = d['validation']['metrics']
    old = met.get('max_abs_library_correlation')
    met['max_abs_library_correlation'] = round(float(max_pear[l]), 4)
    met['max_abs_library_correlation_vs'] = str(best_pear[l])
    met['max_abs_library_correlation_spearman'] = round(float(max_spea[l]), 4)
    met['max_abs_library_correlation_spearman_vs'] = str(best_spea[l])
    met['library_size_at_corr_check'] = len(labels)
    d['validation']['last_validated'] = '2026-07-16'
    d['validation']['corr_check_note'] = (
        f"max_abs_library_correlation recomputed against full {len(labels)}-factor library "
        f"(per-date Pearson mean; spearman stored for reference). Prior value vs 4-factor lib: {old}.")
    with open(path, 'w') as fh:
        json.dump(d, fh, indent=2)
    print(f"  updated {l}: {old} -> {met['max_abs_library_correlation']} (vs {best_pear[l]})")

# ---------------- read-back verification ----------------
print("\n=== READ-BACK VERIFICATION ===")
ok = True
for l in labels:
    d = json.load(open(f"factors/{l}.json"))
    assert d['factor_id'] == l
    met = d['validation']['metrics']
    assert abs(met['ic']) >= 0.007 and abs(met['icir']) >= 0.084, f"gate fail {l}"
    c = met['max_abs_library_correlation']
    if c >= CORR_GATE:
        ok = False
        print(f"  !! {l}: full-library corr {c} >= {CORR_GATE} -> needs review/deprecation")
    else:
        print(f"  OK {l}: ic={met['ic']} icir={met['icir']} full_lib_corr={c} status={d['validation']['status']}")
print("READ-BACK DONE" if ok else "READ-BACK DONE (with flags)")
print(f"total time {time.time()-t0:.0f}s")
