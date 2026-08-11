"""miner_1 round 2026-09-24: screen novel factor ideas (batch A).

Novel territory vs the 18-factor library (beta-heavy: spx/hs300/cn10y/dxy/vix/
eurusd/comm_basket/copper_gold/down_beta + trend/vol family). Candidates here:

  1. xau_beta_60         - rolling beta of asset ret to XAU (gold) ret (60d)   [safe-haven beta]
  2. wti_beta_60         - rolling beta of asset ret to WTI (oil) ret (60d)    [energy beta]
  3. us10y_beta_60       - rolling beta of asset ret to d(US10Y) (60d)         [rates beta]
  4. crypto_beta_60      - rolling beta of asset ret to BTC ret (60d)          [crypto risk appetite]
  5. eth_beta_60         - rolling beta of asset ret to ETH ret (60d)          [alt-coin beta]
  6. realized_kurt_60    - excess kurtosis of 60d daily returns                [tail thickness]
  7. ar1_60              - lag-1 autocorrelation of daily returns (60d)        [serial dependence]
  8. volume_trend_20_60  - 20d mean volume / 60d mean volume                   [volume expansion]
  9. volume_hhi_20       - Herfindahl concentration of volume shares (20d)     [volume concentration]
 10. coexceed_down_60    - joint downside co-exceedance with SPX (60d)         [tail dependence]
 11. disp_beta_60        - beta of asset ret to cross-sectional dispersion (60d) [stress sensitivity]
 12. gap_vol_ratio_20    - std(overnight gap)/std(intraday ret) (20d)          [gap volatility share]
 13. updown_volume_20    - avg volume up-days / avg volume down-days (20d)     [volume-price asym]

Gate: |IC10|>=0.007, |ICIR10|>=0.084, max_abs_library_correlation<0.5.
Validation window 2020-01-01..2026-07-15 (shared warm-up), canonical grid.
"""
import sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, factor_to_panel,
                           validate_factor, signal_matrix, VAL_START, VAL_END)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)}", flush=True)

# ---------------- full effective library rank panels (for rho audit) ----------------
lib_panels = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if art:
            arr = np.load('factors/' + art, allow_pickle=False)
            if arr.shape == (len(grid), len(WATCHLIST)):
                lib_panels[fid] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)

# reconstruct the 3 library factors without artifacts
vix = None
try:
    from factor_common import load_index
    vix = load_index('VIX', prices=prices)
except Exception:
    pass

def f_hilo(df, s):
    return (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min())
def f_vixb(df, s):
    if vix is None:
        return None
    r = df['close'].pct_change(); vr = vix['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
    return (-b * (vix['close'] / vix['close'].shift(20) - 1.0)).reindex(z.index)
def f_vov(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()

for fid, fn in [('hilo_pos_60', f_hilo), ('vix_beta_cond_60x20', f_vixb), ('vol_of_vol20x60', f_vov)]:
    if fid in lib_panels:
        continue
    p = factor_to_panel(fn, prices)
    if len(p):
        lib_panels[fid] = pd.DataFrame(signal_matrix(p, grid), index=grid, columns=WATCHLIST)

print(f"library panels for rho: {len(lib_panels)} -> {sorted(lib_panels.keys())}", flush=True)

def max_lib_corr(panel):
    best, best_id = 0.0, None
    pm = signal_matrix(panel, grid)
    for fid, lp in lib_panels.items():
        lm = lp.values
        corrs = []
        for t in range(len(grid)):
            x = pm[t]; y = lm[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xv = x[m]; yv = y[m]
                xc = xv - xv.mean(); yc = yv - yv.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if den > 0:
                    corrs.append((xc * yc).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---------------- candidate constructions ----------------
spx_ret = prices['SPX']['close'].pct_change()
btc_ret = prices['BTC']['close'].pct_change()
eth_ret = prices['ETH']['close'].pct_change()
xau_ret = prices['XAU']['close'].pct_change()
wti_ret = prices['WTI']['close'].pct_change()
us10y = prices['US10Y']['close']
us10y_chg = us10y.diff()

def make_beta(reg, w=60):
    def f(df, s):
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), reg.rename('x')], axis=1).dropna()
        b = z['r'].rolling(w).cov(z['x']) / z['x'].rolling(w).var()
        return b.reindex(z.index)
    return f

def make_kurt(w):
    def f(df, s):
        r = df['close'].pct_change()
        mu = r.rolling(w).mean()
        sd = r.rolling(w).std()
        k = ((r - mu) ** 4).rolling(w).mean() / (sd ** 4).replace(0, np.nan) - 3.0
        return k
    return f

def make_ar1(w):
    def f(df, s):
        r = df['close'].pct_change()
        return r.rolling(w).corr(r.shift(1))
    return f

def make_vol_trend(ws, wl):
    def f(df, s):
        v = df['volume'].replace(0, np.nan)
        return v.rolling(ws).mean() / v.rolling(wl).mean().replace(0, np.nan)
    return f

def make_vol_hhi(w):
    def f(df, s):
        v = df['volume'].replace(0, np.nan)
        sh = v / v.rolling(w).sum()
        return (sh ** 2).rolling(w).sum()
    return f

def make_coexceed_down(w, thresh=0.01):
    def f(df, s):
        r = df['close'].pct_change()
        joint = (r < -thresh) & (spx_ret < -thresh)
        return joint.rolling(w).mean()
    return f

def make_disp_beta(w):
    # daily cross-sectional dispersion of the 15-asset returns (manual, min 8 obs)
    rets = pd.DataFrame({s: prices[s]['close'].pct_change() for s in WATCHLIST})
    def _cs_std(row):
        x = row.dropna().values
        if len(x) < 8:
            return np.nan
        m = x.mean()
        return float(np.sqrt(((x - m) ** 2).sum() / (len(x) - 1)))
    disp = rets.apply(_cs_std, axis=1)
    def f(df, s):
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), disp.rename('d')], axis=1).dropna()
        b = z['r'].rolling(w).cov(z['d']) / z['d'].rolling(w).var()
        return b.reindex(z.index)
    return f

def make_gap_vol_ratio(w):
    def f(df, s):
        gap = df['open'] / df['close'].shift(1) - 1.0
        intra = df['close'] / df['open'] - 1.0
        gs = gap.rolling(w).std()
        iv = intra.rolling(w).std()
        return (gs / iv.replace(0, np.nan))
    return f

def make_updown_volume(w):
    def f(df, s):
        r = df['close'].pct_change()
        v = df['volume'].replace(0, np.nan)
        up = v.where(r > 0)
        dn = v.where(r < 0)
        return up.rolling(w).mean() / dn.rolling(w).mean().replace(0, np.nan)
    return f

cands = {
    'xau_beta_60':        (make_beta(xau_ret, 60), 'beta of asset ret to XAU ret (60d)', 'safe-haven beta'),
    'wti_beta_60':        (make_beta(wti_ret, 60), 'beta of asset ret to WTI ret (60d)', 'energy beta'),
    'us10y_beta_60':      (make_beta(us10y_chg, 60), 'beta of asset ret to d(US10Y) (60d)', 'rates beta'),
    'crypto_beta_60':     (make_beta(btc_ret, 60), 'beta of asset ret to BTC ret (60d)', 'crypto risk appetite'),
    'eth_beta_60':        (make_beta(eth_ret, 60), 'beta of asset ret to ETH ret (60d)', 'alt-coin beta'),
    'realized_kurt_60':   (make_kurt(60), 'excess kurtosis of 60d daily returns', 'tail thickness'),
    'ar1_60':             (make_ar1(60), 'lag-1 autocorrelation of daily returns (60d)', 'serial dependence'),
    'volume_trend_20_60': (make_vol_trend(20, 60), '20d avg volume / 60d avg volume', 'volume expansion'),
    'volume_hhi_20':      (make_vol_hhi(20), 'Herfindahl of 20d daily volume shares', 'volume concentration'),
    'coexceed_down_60':   (make_coexceed_down(60), 'joint down co-exceedance with SPX (60d)', 'tail dependence'),
    'disp_beta_60':       (make_disp_beta(60), 'beta of asset ret to cross-sectional dispersion (60d)', 'stress sensitivity'),
    'gap_vol_ratio_20':   (make_gap_vol_ratio(20), 'std(overnight gap)/std(intraday ret) (20d)', 'gap volatility share'),
    'updown_volume_20':   (make_updown_volume(20), 'avg volume up-days / avg volume down-days (20d)', 'volume-price asymmetry'),
}

results = {}
for fid, (fn, desc, tag) in cands.items():
    panel = factor_to_panel(fn, prices)
    if panel is None or len(panel) == 0:
        print(f"{fid}: EMPTY panel -> skip", flush=True)
        continue
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None", flush=True)
        continue
    rho, rho_id = max_lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'ok': bool(ok), 'metrics': {k: v for k, v in m.items() if k != '_rank_matrix'},
                    'desc': desc, 'tag': tag}
    dec = {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()}
    print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"coverage={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rho_id})", flush=True)
    print(f"  decay: {dec}", flush=True)
    # regime ICs on the 10d IC series
    ic10 = None
    fwd10 = None
    from factor_common import forward_returns, rank_ic_series
    fwd10 = forward_returns(prices, 10)
    ic10 = rank_ic_series(panel, fwd10, 8)
    ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    for nm, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                     ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                     ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic10[(ic10.index >= a) & (ic10.index <= b)]
        print(f"  {nm}: {sub.mean():.4f} (n={len(sub)})", flush=True)
    rec = ic10[ic10.index >= '2025-07-16']
    if len(rec) > 30:
        r_ic = rec.mean(); r_icir = r_ic / rec.std(ddof=1) if rec.std(ddof=1) > 0 else 0.0
        print(f"  recent_1y: ic={r_ic:.4f} icir={r_icir:.4f} (n={len(rec)})", flush=True)
    print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007, |ICIR|={abs(m['icir']):.4f}/0.084, rho={rho:.3f}/0.5)", flush=True)

with open('scripts/miner_1_20260924_results_batchA.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)
print("\n=== SUMMARY ===")
for fid, r in results.items():
    m = r['metrics']
    print(f"{fid:22s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m['max_abs_library_correlation']:.3f} ({m.get('max_corr_library_id')})")
