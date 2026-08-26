"""miner_1 cycle 2033-09-29. Visible through 2033-09-28 (prev completed trading day).
Re-validate current effective library factors on recent window AND explore fresh
candidate families. Cross-sectional IC on 15-asset universe.
Gates: abs IC>=0.0070 and abs ICIR>=0.084 at 10d horizon; >=8 valid names/date.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner_1_lib import build_panel, load_macro, compute_ic, coverage, turnover, decay_ic

VIS = pd.Timestamp('2033-09-28')

closes, highs, lows, vols, rets = build_panel(end=VIS)
idx = closes.index
print(f"Panel: {closes.shape[0]} dates x {closes.shape[1]} assets, {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d}", flush=True)

vix = load_macro('VIX', VIS); dxy = load_macro('DXY', VIS); cny = load_macro('USDCNY', VIS)
jpy = load_macro('USDJPY', VIS); eur = load_macro('EURUSD', VIS)
dVIX = vix.pct_change(); dDXY = dxy.pct_change(); dCNY = cny.pct_change()
dJPY = jpy.pct_change(); dEUR = eur.pct_change()

def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5, fwd10, fwd20 = fwd(5), fwd(10), fwd(20)

def beta_win(x, y, w):
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for c in x.columns:
        out[c] = x[c].rolling(w).cov(y)
    out = out.div(y.rolling(w).var())
    return out

def corr_win(x, y, w):
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for c in x.columns:
        out[c] = x[c].rolling(w).corr(y)
    return out

F = {}
# Macro-beta family
F['vix_beta_neg20'] = -beta_win(rets, dVIX, 20)
F['vix_beta_neg40'] = -beta_win(rets, dVIX, 40)
F['dxy_beta_neg20'] = -beta_win(rets, dDXY, 20)
F['jpy_beta_20'] = beta_win(rets, dJPY, 20)
j60 = beta_win(rets, dJPY, 60)
F['jpy_beta_chg_20_60'] = beta_win(rets, dJPY, 20) - j60
F['eur_beta_20'] = beta_win(rets, dEUR, 20)
# Vol timing
F['vol_ratio_5_60'] = -rets.rolling(5).std().div(rets.rolling(60).std())
F['rng_pos_neg20'] = -((closes - lows.rolling(20).min()) / (highs.rolling(20).max() - lows.rolling(20).min()))
# Momentum
F['mom_20_neg'] = -closes.pct_change(20)
F['mom_60_neg'] = -closes.pct_change(60)
F['ret_252'] = closes.pct_change(252)
F['mom_10_20diff'] = closes.pct_change(10) - closes.pct_change(20)
# Time-series
F['close_vs_ma20'] = closes.div(closes.rolling(20).mean()) - 1.0
F['close_vs_ma60'] = closes.div(closes.rolling(60).mean()) - 1.0
F['dist_52w_high'] = closes.div(closes.rolling(252).max()) - 1.0
F['drawdown_60'] = closes.div(closes.rolling(60).max()) - 1.0
F['max_dd_20'] = -closes.rolling(20).max().div(closes) + 1.0
# Cross-asset corr
F['corr_spx_neg20'] = -corr_win(rets, rets['SPX'], 20)
F['corr_spx_chg_20_60'] = corr_win(rets, rets['SPX'], 20) - corr_win(rets, rets['SPX'], 60)
F['corr_xau_20'] = corr_win(rets, rets['XAU'], 20)

print("\n===== CANDIDATE SWEEP FULL (2020..2033-09-28) =====", flush=True)
results = {}
for name, fv in F.items():
    a = compute_ic(fv, fwd10)
    b = compute_ic(fv, fwd5); c = compute_ic(fv, fwd20)
    ok = abs(a['IC']) >= 0.0070 and abs(a['ICIR']) >= 0.084
    results[name] = (a, ok, fv)
    print(f"[{'OK' if ok else '--'}] {name:24s} IC={a['IC']:+.4f} ICIR={a['ICIR']:+.4f} n={a['n']:4d} hit={a['hit']:.3f} | [5]{b['IC']:+.3f}[20]{c['IC']:+.3f}", flush=True)

print("\n===== PASSERS: RECENT 2Y + 1Y + DECAY =====", flush=True)
s2y = pd.Timestamp('2031-09-28'); s1y = pd.Timestamp('2032-09-28')
for name, (a, ok, fv) in results.items():
    if ok:
        f = fv.reindex(fwd10.index)
        f = f.loc[f.index >= s2y]
        r2 = compute_ic(f, fwd10.loc[fwd10.index >= s2y])
        r1 = compute_ic(f.loc[f.index >= s1y], fwd10.loc[fwd10.index >= s1y])
        cov, d8 = coverage(f)
        to = turnover(f)
        dec = decay_ic(fv, rets)
        print(f"\n{name}: full IC={a['IC']:+.4f} ICIR={a['ICIR']:+.4f} n={a['n']} hit={a['hit']:.3f}", flush=True)
        print(f"  recent2y IC={r2['IC']:+.4f} ICIR={r2['ICIR']:+.4f} n={r2['n']} | 1y IC={r1['IC']:+.4f} ICIR={r1['ICIR']:+.4f} n={r1['n']}", flush=True)
        print(f"  cov={cov:.3f} dates_ge8={d8:.3f} turn={to:.3f} decay={ {k: round(v,3) for k,v in dec.items()} }", flush=True)

print("\nDONE", flush=True)