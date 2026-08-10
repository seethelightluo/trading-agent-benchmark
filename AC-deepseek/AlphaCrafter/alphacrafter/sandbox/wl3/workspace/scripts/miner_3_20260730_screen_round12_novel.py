"""Round-12 screen: novel factor batch for miner_3.

Candidates (all interpretable, distinct from the 13-factor library and the ~75
previously tested IDs):
 1. obv_slope_20          - OBV 20d slope normalized by 20d mean volume (volume-flow trend)
 2. mfi_14                - Money Flow Index (14d oscillator)
 3. cci_20                - Commodity Channel Index (20d)
 4. vol_imbalance_20      - 20d signed volume imbalance (up-vol minus down-vol share)
 5. mkt_beta_60           - 60d rolling beta of asset returns to equal-weight cross-asset basket
 6. usdcny_beta_cond_60x20 - conditional beta to USDCNY changes * 20d USDCNY move
 7. price_volume_corr_20  - rolling 20d corr(daily return, volume pct change)
 8. atr_pct_20            - ATR(20)/close normalized volatility

Validation: shared factor_common battery (daily Spearman IC vs 10d fwd return,
2020-01-01..2026-07-15), max-abs pairwise rho vs ALL effective library signal
artifacts (recomputing the 3 without .npy). Admission: |IC|>=0.007, |ICIR|>=0.084,
rho<0.5.
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)

# ---------- library artifacts (all EFFECTIVE json + .npy; recompute missing) ----------
lib = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            art = d.get('signal_artifact')
            if art and Path('factors', art).exists():
                lib[d['factor_id']] = np.load(Path('factors', art))
            else:
                lib[d['factor_id']] = None  # mark for recompute
    except Exception as e:
        print("lib skip", f, e)

# recompute missing library panels from definitions
def f_hilo60(df, s):
    hi = df['high'].rolling(60).max(); lo = df['low'].rolling(60).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)

def f_vixbeta(df, s, vix=None):
    if vix is None: return None
    r = df['close'].pct_change(); vr = vix['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
    return (-b * (vix['close'] / vix['close'].shift(20) - 1.0)).reindex(z.index)

def f_vov(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()

vix = load_index('VIX', prices=prices)
for fid, fn in [('hilo_pos_60', f_hilo60), ('vol_of_vol20x60', f_vov)]:
    if lib.get(fid) is None:
        p = factor_to_panel(fn, prices)
        lib[fid] = signal_matrix(p, grid)
        print(f"recomputed lib {fid}: panel {p.shape}", flush=True)
if lib.get('vix_beta_cond_60x20') is None:
    p = factor_to_panel(lambda df, s: f_vixbeta(df, s, vix), prices)
    lib['vix_beta_cond_60x20'] = signal_matrix(p, grid)
    print(f"recomputed lib vix_beta_cond_60x20: panel {p.shape}", flush=True)
print(f"library artifacts: {len(lib)} -> {sorted(lib)}", flush=True)

MIN_V = 8


def rank_rows(M):
    T, n = M.shape
    R = np.full_like(M, np.nan)
    for t in range(T):
        v = M[t]
        m = np.isfinite(v)
        if m.sum() >= MIN_V:
            idx = np.where(m)[0]
            R[t, idx] = v[idx].argsort().argsort().astype(float)
    return R


def row_spearman(RA, RB):
    m = np.isfinite(RA) & np.isfinite(RB)
    A = np.where(m, RA, np.nan)
    B = np.where(m, RB, np.nan)
    cnt = m.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        Ac = A - np.nanmean(A, axis=1, keepdims=True)
        Bc = B - np.nanmean(B, axis=1, keepdims=True)
        num = np.nansum(Ac * Bc, axis=1)
        den = np.sqrt(np.nansum(Ac * Ac, axis=1) * np.nansum(Bc * Bc, axis=1))
        rho = num / den
    rho[~((cnt >= MIN_V) & (den > 0))] = np.nan
    return rho


def max_lib_corr(mat):
    Rc = rank_rows(mat)
    best, best_id = 0.0, None
    for fid, la in lib.items():
        if la is None:
            continue
        if la.shape[0] != mat.shape[0]:
            Rc_use = Rc[-la.shape[0]:]
        else:
            Rc_use = Rc
        Rl = rank_rows(la)
        rho = row_spearman(Rc_use, Rl)
        r = float(np.nanmean(rho)) if np.isfinite(rho).any() else 0.0
        if abs(r) > best:
            best, best_id = abs(r), fid
    return best, best_id


# ---------- candidate definitions ----------
# 1. OBV slope
def obv_slope_20(df, s):
    r = df['close'].pct_change()
    v = df['volume'].astype(float).replace(0, np.nan)
    obv = (np.sign(r) * v).fillna(0.0).cumsum()
    slope = obv.diff(20)
    return slope / v.rolling(20).mean().replace(0, np.nan)


# 2. Money Flow Index
def mfi_14(df, s):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    mf = tp * df['volume'].astype(float)
    d = tp.diff()
    pos = mf.where(d > 0, 0.0).rolling(14).sum()
    neg = mf.where(d < 0, 0.0).rolling(14).sum()
    ratio = pos / neg.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + ratio)


# 3. Commodity Channel Index
def cci_20(df, s):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    sma = tp.rolling(20).mean()
    md = (tp - sma).abs().rolling(20).mean().replace(0, np.nan)
    return (tp - sma) / (0.015 * md)


# 4. Volume imbalance (signed by daily return)
def vol_imbalance_20(df, s):
    r = df['close'].pct_change()
    v = df['volume'].astype(float).replace(0, np.nan)
    signed = np.sign(r) * v
    return signed.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)


# 5. Market beta (60d rolling beta vs equal-weight basket)
_basket_ret = None


def _get_basket_ret():
    global _basket_ret
    if _basket_ret is not None:
        return _basket_ret
    rets = []
    for s, df in prices.items():
        rets.append(df['close'].pct_change().rename(s))
    b = pd.concat(rets, axis=1).mean(axis=1, skipna=True)
    _basket_ret = b
    return b


def mkt_beta_60(df, s):
    b = _get_basket_ret()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), b.rename('b')], axis=1).dropna()
    cov = z['r'].rolling(60).cov(z['b'])
    var = z['b'].rolling(60).var()
    return (cov / var.replace(0, np.nan)).reindex(z.index)


# 6. USDCNY conditional beta
_usdcny = None


def _get_usdcny():
    global _usdcny
    if _usdcny is None:
        _usdcny = load_index('USDCNY', prices=prices)
    return _usdcny


def usdcny_beta_cond_60x20(df, s):
    usdcny = _get_usdcny()
    if usdcny is None:
        return None
    r = df['close'].pct_change()
    vr = usdcny['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
    return (-b * (usdcny['close'] / usdcny['close'].shift(20) - 1.0)).reindex(z.index)


# 7. Price-volume correlation
def price_volume_corr_20(df, s):
    r = df['close'].pct_change()
    v = df['volume'].astype(float).pct_change()
    z = pd.concat([r.rename('r'), v.rename('v')], axis=1)
    return z['r'].rolling(20).corr(z['v'])


# 8. ATR normalized
def atr_pct_20(df, s):
    pc = df['close'].shift(1)
    tr = pd.concat([(df['high'] - df['low']),
                    (df['high'] - pc).abs(),
                    (df['low'] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(20).mean() / df['close'].replace(0, np.nan)


candidates = {
    'obv_slope_20': dict(fn=obv_slope_20, name='OBV 20d slope / volume',
                         expr='slope20(cumsum(sign(ret)*vol))/mean(vol,20)',
                         deps=['close', 'volume'], direction=1),
    'mfi_14': dict(fn=mfi_14, name='Money Flow Index 14d',
                   expr='100 - 100/(1 + MF_pos14/MF_neg14)', deps=['high', 'low', 'close', 'volume'], direction=1),
    'cci_20': dict(fn=cci_20, name='Commodity Channel Index 20d',
                   expr='(TP - SMA(TP,20))/(0.015*mean(|TP-SMA|,20))', deps=['high', 'low', 'close'], direction=1),
    'vol_imbalance_20': dict(fn=vol_imbalance_20, name='Signed volume imbalance 20d',
                             expr='sum(sign(ret)*vol,20)/sum(vol,20)', deps=['close', 'volume'], direction=1),
    'mkt_beta_60': dict(fn=mkt_beta_60, name='Beta vs equal-weight basket 60d',
                        expr='cov(r_asset, r_basket,60)/var(r_basket,60)', deps=['close'], direction=1),
    'usdcny_beta_cond_60x20': dict(fn=usdcny_beta_cond_60x20, name='USDCNY conditional beta',
                                   expr='-beta(r, dUSDCNY,60)*(USDCNY/USDCNY.shift(20)-1)',
                                   deps=['close', 'USDCNY'], direction=1),
    'price_volume_corr_20': dict(fn=price_volume_corr_20, name='Price-volume corr 20d',
                                 expr='corr(pct_change(close), pct_change(vol), 20)',
                                 deps=['close', 'volume'], direction=1),
    'atr_pct_20': dict(fn=atr_pct_20, name='ATR(20) / close',
                       expr='mean(TR,20)/close', deps=['high', 'low', 'close'], direction=1),
}

results = {}
panels = {}
for fid, cfg in candidates.items():
    panel = factor_to_panel(cfg['fn'], prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT -> skip", flush=True)
        results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
        continue
    mat = signal_matrix(panel, grid)
    rho, lib_id = max_lib_corr(mat)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = lib_id
    # recent 1y IC
    ic10 = None
    from factor_common import forward_returns, rank_ic_series
    ic_s = rank_ic_series(panel, forward_returns(prices, 10))
    ic_s = ic_s[(ic_s.index >= pd.Timestamp('2025-07-15')) & (ic_s.index <= pd.Timestamp('2026-07-15'))]
    if len(ic_s) > 30:
        m['recent_1y_ic'] = float(ic_s.mean())
        m['recent_1y_icir'] = float(ic_s.mean() / ic_s.std(ddof=1)) if ic_s.std(ddof=1) > 0 else 0.0
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'ok': ok, 'metrics': m}
    panels[fid] = panel
    print(f"{fid}: IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"rho={rho:.3f}({lib_id}) 1yIC={m.get('recent_1y_ic', float('nan')):+.4f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("   decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}, flush=True)

json.dump(results, open('scripts/miner_3_20260730_results_round12.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20260730_results_round12.json")
