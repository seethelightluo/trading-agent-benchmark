"""Round-14 screen: fixed library correlation audit (panels on current canonical
grid, date-aligned) + round-13 leftovers + new novel candidates.

Motivation: round-13 script crashed because persisted .npy artifacts sit on an
older grid (2335 dates) while the current canonical grid has 2388 dates. We
rebuild all 13 EFFECTIVE library panels on the current grid (definitions in the
JSONs / library_rebuild.py), then audit candidates with date-aligned Spearman.

New novel ideas this batch:
  - usdjpy_beta_cond_60x20: global carry-trade risk (beta to USDJPY * 20d JPY move)
  - yieldspread_beta_60: beta of asset returns to CN10Y-US10Y spread daily change
  - wti_beta_cond_60x20: inflation-commodity risk (beta to WTI * 20d WTI move)
  - autocorr_20: 20d lag-1 autocorrelation of daily returns (regime/reversal)
  - maxdd_60: 60d max drawdown depth
  - ret_skew_20: 20d skew of daily close-to-close returns
  - corr_btc_30: 30d rolling correlation of asset returns with BTC returns
  - beta_asym_60: up-day beta - down-day beta vs SPX
  - vwap_dev_20: (close - 20d VWAP)/close
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor,
                           forward_returns, rank_ic_series, max_library_correlation)
from miner_3_20260730_library_rebuild import build_library_panels

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
vix = load_index('VIX', prices=prices)
dxy = load_index('DXY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
usdjpy = load_index('USDJPY', prices=prices)
print(f"prices {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)
print(f"idx: VIX {False if vix is None else len(vix)} DXY {False if dxy is None else len(dxy)} "
      f"EURUSD {False if eurusd is None else len(eurusd)} USDJPY {False if usdjpy is None else len(usdjpy)}", flush=True)

# ---------- rebuild library panels (13 EFFECTIVE factors) on current grid ----------
lib = build_library_panels(prices, vix, dxy, eurusd)
# factors missing from rebuild helper
spx = prices['SPX']['close']
hs300 = prices['000300.SH']['close']
cn10y = prices['CN10Y']['close']

def f_cn10ybeta(df, s):
    r = df['close'].pct_change()
    rc = cn10y.reindex(df.index).pct_change()
    z = pd.concat([r.rename('r'), rc.rename('c')], axis=1).dropna()
    return z['r'].rolling(60).cov(z['c']) / z['c'].rolling(60).var()

def f_intraskew(df, s):
    return (df['close'] / df['open'] - 1.0).rolling(20).skew()

def f_momaccel(df, s):
    return df['close'].shift(5) / df['close'].shift(65) - df['close'].shift(5) / df['close'].shift(125)

lib['cn10y_beta_60'] = factor_to_panel(f_cn10ybeta, prices)
lib['intraday_ret_skew_20'] = factor_to_panel(f_intraskew, prices)
lib['mom_accel_60_120'] = factor_to_panel(f_momaccel, prices)

# keep only factors currently EFFECTIVE
eff = set()
for f in glob.glob('factors/*.json'):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            eff.add(d['factor_id'])
    except Exception:
        pass
lib = {k: v for k, v in lib.items() if k in eff}
print(f"library panels rebuilt: {len(lib)} -> {sorted(lib)}", flush=True)
for fid, p in lib.items():
    arr = signal_matrix(p, grid)
    print(f"  lib {fid:24s} cov {np.isfinite(arr).mean():.3f}", flush=True)

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
    return (c.shift(1) / c.shift(21) - 1.0 - (xau.shift(1) / xau.shift(21) - 1.0)).reindex(c.index)

def updown_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    up = r[r > 0].rolling(20).std()
    dn = r[r < 0].rolling(20).std()
    return up / dn.replace(0, np.nan)

def usdjpy_beta_cond(df, s):
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    rj = usdjpy['close'].reindex(df.index).pct_change()
    z = pd.concat([r.rename('r'), rj.rename('j')], axis=1).dropna()
    beta = z['r'].rolling(60).cov(z['j']) / z['j'].rolling(60).var()
    mom_j = usdjpy['close'].reindex(z.index) / usdjpy['close'].reindex(z.index).shift(20) - 1.0
    return (beta * mom_j).reindex(z.index)

def yieldspread_beta_60(df, s):
    r = df['close'].pct_change()
    sp = (cn10y - prices['US10Y']['close']).reindex(df.index)
    dsp = sp.diff()
    z = pd.concat([r.rename('r'), dsp.rename('s')], axis=1).dropna()
    return z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()

def wti_beta_cond(df, s):
    r = df['close'].pct_change()
    rw = prices['WTI']['close'].reindex(df.index).pct_change()
    z = pd.concat([r.rename('r'), rw.rename('w')], axis=1).dropna()
    beta = z['r'].rolling(60).cov(z['w']) / z['w'].rolling(60).var()
    mom_w = prices['WTI']['close'].reindex(z.index) / prices['WTI']['close'].reindex(z.index).shift(20) - 1.0
    return (beta * mom_w).reindex(z.index)

def autocorr_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 3 else np.nan, raw=False)

def maxdd_60(df, s):
    c = df['close']
    roll_max = c.rolling(60).max()
    return (c / roll_max - 1.0)

def ret_skew_20(df, s):
    return df['close'].pct_change().rolling(20).skew()

def corr_btc_30(df, s):
    rb = prices['BTC']['close'].reindex(df.index).pct_change()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1)
    return z['r'].rolling(30).corr(z['b'])

def beta_asym_60(df, s):
    r = df['close'].pct_change()
    rs = spx.reindex(df.index).pct_change()
    z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
    up = z[z['s'] > 0]
    dn = z[z['s'] < 0]
    bu = up['r'].rolling(60).cov(up['s']) / up['s'].rolling(60).var()
    bd = dn['r'].rolling(60).cov(dn['s']) / dn['s'].rolling(60).var()
    return (bu - bd).reindex(z.index)

def vwap_dev_20(df, s):
    vwap = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    return (df['close'] - vwap) / vwap.replace(0, np.nan)


candidates = {
    'mfi_7': dict(fn=make_mfi(7), name='Money Flow Index 7d', direction=1),
    'mfi_21': dict(fn=make_mfi(21), name='Money Flow Index 21d', direction=1),
    'cci_7': dict(fn=make_cci(7), name='Commodity Channel Index 7d', direction=1),
    'rev_5': dict(fn=rev_5, name='5d reversal', direction=1),
    'rev_5_vol': dict(fn=rev_5_vol, name='5d reversal / 20d vol', direction=1),
    'corr_xau_60': dict(fn=corr_xau_60, name='60d corr with XAU', direction=1),
    'rel_mom_xau_20': dict(fn=rel_mom_xau_20, name='20d rel momentum vs XAU', direction=1),
    'updown_vol_ratio_20': dict(fn=updown_vol_ratio_20, name='Up/down vol ratio 20d', direction=1),
    'usdjpy_beta_cond_60x20': dict(fn=usdjpy_beta_cond, name='USDJPY conditional beta 60x20', direction=1),
    'yieldspread_beta_60': dict(fn=yieldspread_beta_60, name='Beta to CN10Y-US10Y spread change 60d', direction=1),
    'wti_beta_cond_60x20': dict(fn=wti_beta_cond, name='WTI conditional beta 60x20', direction=1),
    'autocorr_20': dict(fn=autocorr_20, name='20d lag-1 autocorrelation', direction=1),
    'maxdd_60': dict(fn=maxdd_60, name='60d max drawdown depth', direction=1),
    'ret_skew_20': dict(fn=ret_skew_20, name='20d return skewness', direction=1),
    'corr_btc_30': dict(fn=corr_btc_30, name='30d corr with BTC', direction=1),
    'beta_asym_60': dict(fn=beta_asym_60, name='Up-beta minus Down-beta 60d', direction=1),
    'vwap_dev_20': dict(fn=vwap_dev_20, name='20d VWAP deviation', direction=1),
}

fwd10 = forward_returns(prices, 10)
results = {}
for fid, cfg in candidates.items():
    panel = factor_to_panel(cfg['fn'], prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT -> skip", flush=True)
        results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
        continue
    rho, lib_id = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = lib_id
    # sub-period robustness
    ic_s = rank_ic_series(panel, fwd10)
    ic_s = ic_s[(ic_s.index >= pd.Timestamp('2020-01-01')) & (ic_s.index <= pd.Timestamp('2026-07-15'))]
    for name, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                       ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                       ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic_s[(ic_s.index >= pd.Timestamp(a)) & (ic_s.index <= pd.Timestamp(b))]
        m[name] = float(sub.mean()) if len(sub) > 30 else float('nan')
    recent = ic_s[(ic_s.index >= pd.Timestamp('2025-07-15')) & (ic_s.index <= pd.Timestamp('2026-07-15'))]
    if len(recent) > 30:
        m['recent_1y_ic'] = float(recent.mean())
        m['recent_1y_icir'] = float(recent.mean() / recent.std(ddof=1)) if recent.std(ddof=1) > 0 else 0.0
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'ok': ok, 'metrics': m}
    print(f"{fid}: IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({lib_id}) "
          f"p1={m['ic_2020_2022']:+.3f} p2={m['ic_2023_2024']:+.3f} p3={m['ic_2025_2026']:+.3f} "
          f"1y={m.get('recent_1y_ic', float('nan')):+.4f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("   decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}, flush=True)

json.dump(results, open('scripts/miner_3_20260730_results_round14.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20260730_results_round14.json", flush=True)
