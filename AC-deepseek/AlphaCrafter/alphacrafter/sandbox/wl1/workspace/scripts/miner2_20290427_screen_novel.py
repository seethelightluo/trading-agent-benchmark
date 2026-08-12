"""miner2 2029-04-27 (fixed 2029-05-11): broad screen of novel factor families on data through 2029-04-26."""
import sys, json, traceback
import pandas as pd, numpy as np
sys.path.insert(0, 'scripts')
from miner2_factor_val_fast import (load_panel, library_signals, daily_ic_series, ic_metrics,
                                    signal_correlation_matrix, GATE_IC, GATE_ICIR)

panel = load_panel('scripts/panel_cache.pkl')
close = panel['close']; high = panel['high']; low = panel['low']
open_ = panel['open']; ret = panel['ret']; vol = panel['vol']
lnc = np.log(close)
fwd1 = ret.shift(-1)

CUT = '2021-01-01'
mask_idx = close.index >= CUT
close_m = close[mask_idx]; ret_m = ret[mask_idx]; fwd1_m = fwd1[mask_idx]

macro = panel['macro']
vix = macro['VIX']; dxy = macro['DXY']; usdjpy = macro['USDJPY']; eurusd = macro['EURUSD']; usdcny = macro['USDCNY']

# align macro to close index via reindex (forward fill)
def align(s):
    return s.reindex(close.index, method='ffill')
vix_a, dxy_a = align(vix), align(dxy)
usdjpy_a, eurusd_a, usdcny_a = align(usdjpy), align(eurusd), align(usdcny)

rv5 = ret.rolling(5).std(); rv20 = ret.rolling(20).std(); rv60 = ret.rolling(60).std()
sma20 = close.rolling(20).mean(); sma60 = close.rolling(60).mean(); sma120 = close.rolling(120).mean()

# helper: broadcast a common Series to a per-instrument DataFrame for rolling-cov beta calc
def wide(s):
    return pd.DataFrame({c: s.values for c in close.columns}, index=close.index)

F = {}
# --- momentum / trend family ---
for nd in (10, 20, 40, 60, 90, 120, 180, 252):
    F[f'mom_{nd}d'] = close / close.shift(nd) - 1.0
F['mom_120d_skip20'] = close.shift(20) / close.shift(140) - 1.0
F['mom_60d_skip5'] = close.shift(5) / close.shift(65) - 1.0
F['mom_90d_skip10'] = close.shift(10) / close.shift(100) - 1.0
F['trend_20d'] = (close - sma20) / sma20
F['trend_60d'] = (close - sma60) / sma60
F['ma20_ma60'] = sma20 / sma60 - 1.0
F['ma60_ma120'] = sma60 / sma120 - 1.0
# vol-scaled momentum
F['mom60_vs'] = (close / close.shift(60) - 1.0) / rv60
F['mom20_vs'] = (close / close.shift(20) - 1.0) / rv20
F['mom120_vs'] = (close / close.shift(120) - 1.0) / rv60
# --- drawdown / distance from high ---
F['dd_60'] = 1.0 - close / close.rolling(60).max()
F['dd_252'] = 1.0 - close / close.rolling(252).max()
F['dist_252_high'] = close / close.rolling(252).max() - 1.0
# --- volatility family ---
F['inv_vol20'] = -rv20
F['vol20_vol60'] = rv20 / rv60 - 1.0
F['vol_chg_20'] = rv20 / rv20.shift(20) - 1.0
F['vol_chg_60'] = rv60 / rv60.shift(60) - 1.0
F['range_20'] = (high.rolling(20).max() - low.rolling(20).min()) / close
# --- macro-beta / conditional (broadcast macro to wide frames) ---
vix_ret = vix_a.pct_change(); dxy_ret = dxy_a.pct_change(); usdjpy_ret = usdjpy_a.pct_change()
vix_ret_w, dxy_ret_w, jpy_ret_w = wide(vix_ret), wide(dxy_ret), wide(usdjpy_ret)
beta_vix60 = ret.rolling(60).cov(vix_ret_w) / vix_ret_w.rolling(60).var()
beta_dxy60 = ret.rolling(60).cov(dxy_ret_w) / dxy_ret_w.rolling(60).var()
beta_jpy60 = ret.rolling(60).cov(jpy_ret_w) / jpy_ret_w.rolling(60).var()
F['beta_vix_60'] = -beta_vix60
F['beta_dxy_60'] = -beta_dxy60
F['beta_jpy_60'] = beta_jpy60
# conditional: low-vol momentum (only meaningful when vol below median)
lowvol_mask = (rv20 <= rv20.rolling(120).median()).astype(float)
F['mom60_lowvol_cond'] = (close / close.shift(60) - 1.0) * lowvol_mask
# vix-regime trend: trend only when vix falling
vix_fall = (vix_a <= vix_a.shift(20)).astype(float)
F['trend60_vixfall'] = ((close - sma60) / sma60) * vix_fall.to_numpy()[:, None]
# vix-level conditional reversal: reversal strongest when vix elevated
vix_hi = (vix_a > vix_a.rolling(60).median()).astype(float)
F['rev2d_vixhi'] = -(lnc - lnc.shift(2)) * vix_hi.to_numpy()[:, None]
# --- cross-sectional relative strength ---
xs_mean = close.mean(axis=1)
F['rel_mom20'] = (close / close.shift(20) - 1.0) - (xs_mean / xs_mean.shift(20) - 1.0).to_numpy()[:, None]
F['rel_mom60'] = (close / close.shift(60) - 1.0) - (xs_mean / xs_mean.shift(60) - 1.0).to_numpy()[:, None]
# --- rate-carry cross-asset: sensitivity to US10Y change (5d), per instrument ---
us10y_w = wide(close['US10Y'].pct_change(5))
F['beta_us10y5_60'] = ret.rolling(60).cov(us10y_w) / us10y_w.rolling(60).var()
# --- dollar-beta: sensitivity to DXY 20d change ---
F['beta_dxy20_60'] = ret.rolling(60).cov(dxy_ret_w) / dxy_ret_w.rolling(60).var()

lib = library_signals(panel)
results = {}
for name, s in F.items():
    try:
        sm = s[mask_idx]
        if sm.notna().sum().sum() < 100:
            print(f"[{name}] SKIP (insufficient data {int(sm.notna().sum().sum())})")
            continue
        ic_s = daily_ic_series(sm, fwd1_m)
        m = ic_metrics(ic_s, sm, fwd1_m, label=name)
        maxabs, rows = signal_correlation_matrix(sm, lib)
        m['max_abs_library_correlation'] = maxabs
        m['gate'] = bool(abs(m['ic']) >= GATE_IC and abs(m['icir']) >= GATE_ICIR)
        results[name] = m
        r6 = m.get('recent_6m', {}); y29 = m.get('by_year', {}).get('2029', {})
        print(f"[{name:22s}] IC={m['ic']:+.4f} ICIR={m['icir']:+.3f} hit={m['hit']:.3f} cov={m['coverage']:.3f} "
              f"n={m['n_dates']} | 2029IC={y29.get('ic', float('nan')):+.4f} 6mIC={r6.get('ic', float('nan')):+.4f} "
              f"maxcorr={maxabs:.2f} GATE={'PASS' if m['gate'] else 'fail'}")
    except Exception as e:
        print(f"[{name}] ERROR: {e}")
        traceback.print_exc()

with open('scripts/miner2_20290427_screen_novel.json', 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'by_year'} for k, v in results.items()}, f, indent=1, default=float)
print("\nsaved scripts/miner2_20290427_screen_novel.json")
