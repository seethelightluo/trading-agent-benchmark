"""Round-13 screen: variant + novel factor batch (batch 2).

Motivation from round 12: MFI/CCI passed IC/ICIR gates but rho>0.5 vs hilo_pos_60.
Test shorter/longer windows to break the range-position correlation, plus novel
directions: short-term reversal (classic cross-sectional), gold-haven correlation,
relative momentum vs XAU, and 20d upside/downside vol asymmetry.
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor,
                           forward_returns, rank_ic_series)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)

# ---------- library artifacts (EFFECTIVE json + .npy; recompute missing) ----------
lib = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            art = d.get('signal_artifact')
            if art and Path('factors', art).exists():
                lib[d['factor_id']] = np.load(Path('factors', art))
            else:
                lib[d['factor_id']] = None
    except Exception as e:
        print("lib skip", f, e)


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
        lib[fid] = signal_matrix(factor_to_panel(fn, prices), grid)
if lib.get('vix_beta_cond_60x20') is None:
    lib['vix_beta_cond_60x20'] = signal_matrix(
        factor_to_panel(lambda df, s: f_vixbeta(df, s, vix), prices), grid)
print(f"library artifacts: {len(lib)}", flush=True)

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
        Rl = rank_rows(la)
        rho = row_spearman(Rc, Rl)
        r = float(np.nanmean(rho)) if np.isfinite(rho).any() else 0.0
        if abs(r) > best:
            best, best_id = abs(r), fid
    return best, best_id


# ---------- candidates ----------
def make_mfi(w):
    def mfi_w(df, s):
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        mf = tp * df['volume'].astype(float)
        d = tp.diff()
        pos = mf.where(d > 0, 0.0).rolling(w).sum()
        neg = mf.where(d < 0, 0.0).rolling(w).sum()
        ratio = pos / neg.replace(0, np.nan)
        return 100.0 - 100.0 / (1.0 + ratio)
    return mfi_w


def make_cci(w):
    def cci_w(df, s):
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        sma = tp.rolling(w).mean()
        md = (tp - sma).abs().rolling(w).mean().replace(0, np.nan)
        return (tp - sma) / (0.015 * md)
    return cci_w


def rev_5(df, s):
    return -(df['close'].shift(1) / df['close'].shift(6) - 1.0)


def rev_5_vol(df, s):
    c = df['close']
    r5 = -(c.shift(1) / c.shift(6) - 1.0)
    vol20 = c.pct_change().rolling(20).std()
    return r5 / vol20.replace(0, np.nan)


_xau_ret = None


def _get_xau_ret():
    global _xau_ret
    if _xau_ret is None:
        _xau_ret = prices['XAU']['close'].pct_change()
    return _xau_ret


def corr_xau_60(df, s):
    xr = _get_xau_ret()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), xr.rename('x')], axis=1)
    return z['r'].rolling(60).corr(z['x'])


def rel_mom_xau_20(df, s):
    c = df['close']
    xau = prices['XAU']['close']
    mom_a = c.shift(1) / c.shift(21) - 1.0
    mom_x = xau.shift(1) / xau.shift(21) - 1.0
    return (mom_a - mom_x).reindex(c.index)


def updown_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    up = r[r > 0].rolling(20).std()
    dn = r[r < 0].rolling(20).std()
    return up / dn.replace(0, np.nan)


candidates = {
    'mfi_7': dict(fn=make_mfi(7), name='Money Flow Index 7d',
                  expr='100-100/(1+MF_pos7/MF_neg7)', deps=['high', 'low', 'close', 'volume'], direction=1),
    'mfi_21': dict(fn=make_mfi(21), name='Money Flow Index 21d',
                   expr='100-100/(1+MF_pos21/MF_neg21)', deps=['high', 'low', 'close', 'volume'], direction=1),
    'cci_7': dict(fn=make_cci(7), name='Commodity Channel Index 7d',
                  expr='(TP-SMA(TP,7))/(0.015*mean(|TP-SMA|,7))', deps=['high', 'low', 'close'], direction=1),
    'rev_5': dict(fn=rev_5, name='5d reversal',
                  expr='-(close.shift(1)/close.shift(6)-1)', deps=['close'], direction=1),
    'rev_5_vol': dict(fn=rev_5_vol, name='5d reversal / 20d vol',
                      expr='-mom5/STD20(ret)', deps=['close'], direction=1),
    'corr_xau_60': dict(fn=corr_xau_60, name='60d corr with XAU returns',
                        expr='corr(r_asset, r_XAU, 60)', deps=['close', 'XAU'], direction=1),
    'rel_mom_xau_20': dict(fn=rel_mom_xau_20, name='20d rel momentum vs XAU',
                           expr='mom20_asset - mom20_XAU', deps=['close', 'XAU'], direction=1),
    'updown_vol_ratio_20': dict(fn=updown_vol_ratio_20, name='Up/down vol ratio 20d',
                                expr='STD(ret|ret>0,20)/STD(ret|ret<0,20)', deps=['close'], direction=1),
}

results = {}
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
    ic_s = rank_ic_series(panel, forward_returns(prices, 10))
    ic_s = ic_s[(ic_s.index >= pd.Timestamp('2025-07-15')) & (ic_s.index <= pd.Timestamp('2026-07-15'))]
    if len(ic_s) > 30:
        m['recent_1y_ic'] = float(ic_s.mean())
        m['recent_1y_icir'] = float(ic_s.mean() / ic_s.std(ddof=1)) if ic_s.std(ddof=1) > 0 else 0.0
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'ok': ok, 'metrics': m}
    print(f"{fid}: IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"rho={rho:.3f}({lib_id}) 1yIC={m.get('recent_1y_ic', float('nan')):+.4f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("   decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}, flush=True)

json.dump(results, open('scripts/miner_3_20260730_results_round13.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20260730_results_round13.json")
