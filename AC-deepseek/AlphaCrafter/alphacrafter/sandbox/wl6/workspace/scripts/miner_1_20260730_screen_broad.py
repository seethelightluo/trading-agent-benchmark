"""Broad screen (fast): horizon-10 daily rank IC only. Detailed metrics for top candidates later."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner_1_20260730_validation_lib import (load_close_panel, load_macro_panel,
    forward_returns, IC_TH, ICIR_TH)

panel = load_close_panel()
macro = load_macro_panel()
fwd10 = forward_returns(panel, (10,))[10]
ret = panel.pct_change()

dxy = macro['DXY']; vix = macro['VIX']; jpy = macro['USDJPY']
cny = macro['USDCNY']; eur = macro['EURUSD']
dxy_r = dxy.pct_change(); vix_r = vix.pct_change()
jpy_r = jpy.pct_change(); cny_r = cny.pct_change(); eur_r = eur.pct_change()

def beta_of(a, m, win):
    return a.rolling(win).cov(m) / m.rolling(win).var()

C = {}
for lb in (10, 20, 40, 60, 90, 120, 180, 250):
    C[f'mom_{lb}d_skip5'] = panel.shift(5) / panel.shift(5 + lb) - 1.0
for lb in (20, 60, 120):
    C[f'risk_adj_mom_{lb}d'] = (panel.shift(5) / panel.shift(5 + lb) - 1.0) / ret.rolling(lb).std()
C['mom_20d_5_60_ratio'] = ((panel.shift(5)/panel.shift(25)-1.0) /
                           (panel.shift(5)/panel.shift(65)-1.0))
for w in (20, 60, 120, 250):
    C[f'dist_sma_{w}d'] = panel / panel.rolling(w).mean() - 1.0
C['price_vs_52w_high'] = panel / panel.rolling(250).max() - 1.0
C['price_vs_52w_low'] = panel / panel.rolling(250).min() - 1.0
C['zscore_5d'] = (panel / panel.shift(5) - 1.0 - ret.rolling(20).mean()) / ret.rolling(20).std()
vol20 = ret.rolling(20).std(); vol60 = ret.rolling(60).std()
C['inv_vol_20d'] = -vol20
C['vol_of_vol20x60'] = vol20.rolling(60).std()
C['vol_ratio_20x60'] = vol20 / vol60
C['vol_zscore_20_250'] = (vol20 - vol60.rolling(5).mean()) / vol60.rolling(5).std()
C['skew_60d'] = ret.rolling(60).skew()
C['kurt_60d'] = ret.rolling(60).kurt()
C['downside_vol_ratio_20x60'] = ret.clip(upper=0).rolling(20).std() / ret.clip(upper=0).rolling(60).std()
C['amihud_20d'] = (ret.abs() / panel).rolling(20).mean()
C['dxy_beta_60d'] = beta_of(ret, dxy_r, 60)
C['dxy_beta_120d'] = beta_of(ret, dxy_r, 120)
C['jpy_beta_60d'] = beta_of(ret, jpy_r, 60)
C['cny_beta_60d'] = beta_of(ret, cny_r, 60)
C['eur_beta_60d'] = beta_of(ret, eur_r, 60)
C['vix_beta_60d'] = beta_of(ret, vix_r, 60)
C['dxy_cond_60x20'] = -beta_of(ret, dxy_r, 60) * (dxy / dxy.shift(20) - 1.0)
C['vix_cond_60x20'] = -beta_of(ret, vix_r, 60) * (vix / vix.shift(20) - 1.0)
C['jpy_cond_60x20'] = -beta_of(ret, jpy_r, 60) * (jpy / jpy.shift(20) - 1.0)
C['cny_cond_60x20'] = -beta_of(ret, cny_r, 60) * (cny / cny.shift(20) - 1.0)
C['us10y_beta_60d'] = beta_of(ret, panel['US10Y'].pct_change(), 60)
C['cn10y_beta_60d'] = beta_of(ret, panel['CN10Y'].pct_change(), 60)
C['wti_copper_rel_20d'] = (panel['WTI'].pct_change(20) - panel['COPPER'].pct_change(20))
C['max_drawdown_60d'] = (panel / panel.rolling(60).max() - 1.0)
C['mom_dispersion_20d'] = panel.pct_change(20).sub(panel.pct_change(20).mean(axis=1), axis=0).abs().mean(axis=1)

def fast_ic(f, fr):
    f = f.reindex(fr.index)
    if isinstance(f, pd.Series):
        f = f.to_frame(name='f0')
    ics = []
    for d in fr.index:
        row = pd.concat([f.loc[d].rename('f'), fr.loc[d].rename('r')], axis=1).dropna()
        if len(row) >= 8:
            ic = row['f'].rank().corr(row['r'].rank())
            if np.isfinite(ic):
                ics.append(ic)
    return np.array(ics)

print(f"{'factor':<26}{'ic':>9}{'icir':>8}{'hit':>7}{'n':>6}  gate")
rows = []
for name, f in C.items():
    if f is None:
        continue
    ics = fast_ic(f, fwd10)
    if len(ics) == 0:
        continue
    ic = float(ics.mean()); icir = ic / (float(ics.std(ddof=1)) or np.nan)
    hit = float((ics > 0).mean())
    flag = 'PASS' if (abs(ic) >= IC_TH and abs(icir) >= ICIR_TH) else ''
    print(f"{name:<26}{ic:>9.4f}{icir:>8.4f}{hit:>7.3f}{len(ics):>6d}  {flag}")
    rows.append((name, abs(ic), abs(icir), ic, icir, flag))

print('\n--- top 12 by |IC|*|ICIR| ---')
for name, aic, aicir, ic, icir, flag in sorted(rows, key=lambda r: r[1]*r[2], reverse=True)[:12]:
    print(f"{name:<26}{ic:>8.4f}{icir:>8.4f}  {flag}")
