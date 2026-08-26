"""miner_1 cycle 2033-08-18. Visible through 2033-08-17. No lookahead.
Explore fresh candidate factor families + cross-sectional IC on 15-asset universe.
Gates: abs IC>=0.0070 and abs ICIR>=0.084 at 10d horizon; >=8 valid names/date.
All candidates are full-panel signals (no per-asset direction flip) so a single
expected direction applies across the 15-asset cross-section.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner_1_lib import (build_panel, load_macro, compute_ic, coverage, turnover,
                         decay_ic, report)

closes, highs, lows, vols, rets = build_panel()
idx = closes.index
print(f"Panel: {closes.shape[0]} dates x {closes.shape[1]} assets, {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d}", flush=True)

vix = load_macro('VIX'); dxy = load_macro('DXY'); cny = load_macro('USDCNY')
jpy = load_macro('USDJPY'); eur = load_macro('EURUSD')
dVIX = vix.pct_change(); dDXY = dxy.pct_change(); dCNY = cny.pct_change()
dJPY = jpy.pct_change(); dEUR = eur.pct_change()

def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5, fwd10, fwd20 = fwd(5), fwd(10), fwd(20)

def rankz(x):
    return x.rank(axis=1, pct=True).sub(0.5)

def corr_win(x, y, w):
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for c in x.columns:
        out[c] = x[c].rolling(w).corr(y)
    return out

def beta_win(x, y, w):
    return x.rolling(w).cov(y).div(y.rolling(w).var())

def roll_corr_ts(s, y, w):
    return s.rolling(w).corr(y)

F = {}

# ---- Macro-beta family (VIX exposures) ----
F['vix_beta_neg20'] = -beta_win(rets, dVIX, 20)          # negative short-horizon VIX beta
F['vix_beta_neg40'] = -beta_win(rets, dVIX, 40)
# ---- Macro-beta family (DXY, CNY, JPY, EUR) ----
F['dxy_beta_neg20'] = -beta_win(rets, dDXY, 20)
F['dxy_beta_neg40'] = -beta_win(rets, dDXY, 40)
F['jpy_beta_20'] = beta_win(rets, dJPY, 20)              # JPY carry/risk proxy
jpy60 = beta_win(rets, dJPY, 60)
F['jpy_beta_chg_20_60'] = beta_win(rets, dJPY, 20) - jpy60
F['eur_beta_20'] = beta_win(rets, dEUR, 20)
# ---- Volatility-timing family ----
F['vol_pct_20'] = -rets.rolling(20).std()                # low short-term vol
F['vol_ratio_5_60'] = -rets.rolling(5).std().div(rets.rolling(60).std())  # vol compression
F['vol_rank_60_20'] = rankz(rets.rolling(20).std().rank(axis=1, pct=True))  # low rel vol vs cross-section
F['rng_pos_neg20'] = -((closes - lows.rolling(20).min()) / (highs.rolling(20).max() - lows.rolling(20).min()))
# ---- Momentum refinements ----
F['mom_20_neg'] = -ret_20 = -closes.pct_change(20)
F['mom_60_neg'] = -closes.pct_change(60)
F['mom_10_20diff'] = closes.pct_change(10) - closes.pct_change(20)   # short>long: fresh acceleration
F['ret_252'] = closes.pct_change(252)                                # 12m trend
# ---- Time-series (close vs past) ----
F['close_vs_ma20'] = closes.div(closes.rolling(20).mean()) - 1.0
F['close_vs_ma60'] = closes.div(closes.rolling(60).mean()) - 1.0
F['dist_52w_high'] = closes.div(closes.rolling(252).max()) - 1.0
F['dist_52w_low'] = closes.div(closes.rolling(252).min()) - 1.0
F['drawdown_60'] = (closes.div(closes.rolling(60).max()) - 1.0)
# ---- Cross-asset correlations / dispersion ----
F['corr_spx_neg20'] = -corr_win(rets, rets['SPX'], 20)
F['corr_xau_20'] = corr_win(rets, rets['XAU'], 20)
F['corr_spx_chg_20_60'] = corr_win(rets, rets['SPX'], 20) - corr_win(rets, rets['SPX'], 60)
F['cny_beta_neg20'] = -beta_win(rets, dCNY, 20)
# ---- Tail/cycle ----
F['max_dd_20'] = -closes.rolling(20).max().div(closes) + 1.0   # recent drawdown depth (contrarian)
# ---- Volume (liquidity) ----
F['vol_neg'] = -vols.rolling(20).mean() if vols.notna().sum().sum() > 0 else None
F['vol_ratio_5_60'] = vols.rolling(5).mean().div(vols.rolling(60).mean())  # volume surge
F = {k: v for k, v in F.items() if v is not None}

print("\n===== CANDIDATE SWEEP (full window through 2033-08-17) =====", flush=True)
results = {}
for name, fv in F.items():
    a, ok = report(name, fv, fwd5, fwd10, fwd20)
    results[name] = (a, ok, fv)

print("\n===== PASSERS: RECENT 2Y (2031-08-18..2033-08-17) + 1Y + DECAY =====", flush=True)
start2y = pd.Timestamp('2031-08-18')
start1y = pd.Timestamp('2032-08-18')
for name, (a, ok, fv) in results.items():
    if ok:
        f = fv.reindex(fwd10.index)
        r2 = compute_ic(f.loc[f.index >= start2y], fwd10.loc[fwd10.index >= start2y])
        r1 = compute_ic(f.loc[f.index >= start1y], fwd10.loc[fwd10.index >= start1y])
        cov, d8 = coverage(f)
        to = turnover(f)
        dec = decay_ic(f, rets)
        print(f"\n{name}: full IC={a['IC']:+.4f} ICIR={a['ICIR']:+.4f} n={a['n']} hit={a['hit']:.3f}", flush=True)
        print(f"  recent2y IC={r2['IC']:+.4f} ICIR={r2['ICIR']:+.4f} n={r2['n']} | 1y IC={r1['IC']:+.4f} ICIR={r1['ICIR']:+.4f} n={r1['n']}", flush=True)
        print(f"  cov={cov:.3f} dates_ge8={d8:.3f} turn={to:.3f} decay={ {k: round(v,3) for k,v in dec.items()} }", flush=True)

print("\nDONE", flush=True)