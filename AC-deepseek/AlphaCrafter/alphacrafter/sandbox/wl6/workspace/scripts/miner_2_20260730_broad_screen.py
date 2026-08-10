"""miner_2 broad screen 2026-07-30: fast horizon-10 rank IC across many factor ideas.

Reuses miner_1's proven data-loading lib (visible_through window respected).
Cross-section = 15 tradables; >=8 valid per date for an IC observation.
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner_1_20260730_validation_lib import (load_close_panel, load_macro_panel,
    forward_returns, IC_TH, ICIR_TH)

panel = load_close_panel()
macro = load_macro_panel()
fwd = forward_returns(panel, (10,))[10]
ret = panel.pct_change()
n_dates = len(panel)
print(f"panel: {n_dates} dates x {panel.shape[1]} assets, "
      f"{panel.index.min().date()} .. {panel.index.max().date()}")

dxy = macro['DXY']; vix = macro['VIX']; jpy = macro['USDJPY']
cny = macro['USDCNY']; eur = macro['EURUSD']
dxy_r = dxy.pct_change(); vix_r = vix.pct_change()
jpy_r = jpy.pct_change(); cny_r = cny.pct_change(); eur_r = eur.pct_change()

def beta_of(a, m, win):
    return a.rolling(win).cov(m) / m.rolling(win).var()

def to_panel(f, idx):
    """Series -> single-column DataFrame aligned to idx."""
    if isinstance(f, pd.Series):
        f = f.to_frame(name=f.name or 'f0')
    return f.reindex(idx)

C = {}
# --- momentum / trend ---
for lb in (5, 10, 20, 30, 60, 90):
    C[f'mom_{lb}d_skip3'] = panel.shift(3) / panel.shift(3 + lb) - 1.0
C['mom_ratio_10_60_skip5'] = (panel.shift(5)/panel.shift(15)-1.0) / (panel.shift(5)/panel.shift(65)-1.0)
C['trend_ema_ratio_20_60'] = panel.ewm(span=20).mean() / panel.ewm(span=60).mean() - 1.0
# --- reversal / mean reversion ---
C['rev_1d'] = -(panel.shift(1) / panel.shift(2) - 1.0)
C['rev_5d'] = -(panel.shift(1) / panel.shift(6) - 1.0)
# --- volatility family ---
vol20 = ret.rolling(20).std(); vol60 = ret.rolling(60).std(); vol10 = ret.rolling(10).std()
C['vol_ratio_10_60'] = vol10 / vol60
C['vol_ratio_20_60'] = vol20 / vol60
C['vol_zscore_20_120'] = (vol20 - vol20.rolling(120).mean()) / vol20.rolling(120).std()
C['downside_vol_20d'] = -ret.clip(upper=0).rolling(20).std()
C['downside_ratio_20_60'] = ret.clip(upper=0).rolling(20).std() / ret.clip(upper=0).rolling(60).std()
C['range_20d'] = (panel.rolling(20).max() - panel.rolling(20).min()) / panel
C['parkinson_20d'] = np.log(panel.rolling(20).max()/panel.rolling(20).min()) / (2*np.sqrt(2*np.log(2)))
# --- distributional ---
C['skew_60d'] = ret.rolling(60).skew()
C['kurt_60d'] = ret.rolling(60).kurt()
C['skew_20d'] = ret.rolling(20).skew()
# --- liquidity / volume ---
C['amihud_20d'] = (ret.abs() / panel).rolling(20).mean()
C['amihud_60d'] = (ret.abs() / panel).rolling(60).mean()
# --- drawdown / distance ---
C['dist_52w_high'] = panel / panel.rolling(250).max() - 1.0
C['dist_52w_low'] = panel / panel.rolling(250).min() - 1.0
C['mdd_60d'] = panel / panel.rolling(60).max() - 1.0
C['recover_20_60'] = (panel / panel.rolling(60).max() - 1.0) - (panel.shift(20) / panel.rolling(60).max().shift(20) - 1.0)
# --- risk-adjusted momentum ---
for lb in (20, 60):
    C[f'risk_adj_mom_{lb}d_skip5'] = (panel.shift(5) / panel.shift(5+lb) - 1.0) / ret.rolling(lb).std()
C['sharpish_60d'] = (ret.rolling(60).mean() / ret.rolling(60).std())
# --- macro beta family ---
C['dxy_beta_60d'] = beta_of(ret, dxy_r, 60)
C['jpy_beta_60d'] = beta_of(ret, jpy_r, 60)
C['eur_beta_60d'] = beta_of(ret, eur_r, 60)
C['cny_beta_60d'] = beta_of(ret, cny_r, 60)
C['vix_beta_60d'] = beta_of(ret, vix_r, 60)
C['dxy_beta_120d'] = beta_of(ret, dxy_r, 120)
# --- conditional macro ---
C['dxy_cond_60x10'] = -beta_of(ret, dxy_r, 60) * (dxy / dxy.shift(10) - 1.0)
C['jpy_cond_60x20'] = -beta_of(ret, jpy_r, 60) * (jpy / jpy.shift(20) - 1.0)
C['cny_cond_60x20'] = -beta_of(ret, cny_r, 60) * (cny / cny.shift(20) - 1.0)
C['eur_cond_60x20'] = -beta_of(ret, eur_r, 60) * (eur / eur.shift(20) - 1.0)
# --- yield curve / cross-asset spreads (asset-level spread series) ---
us10 = panel['US10Y']; cn10 = panel['CN10Y']
C['us10y_mom_20d'] = us10.pct_change(20)
C['cn10y_mom_20d'] = cn10.pct_change(20)
C['yield_spread_20d_chg'] = (us10 - cn10) - (us10.shift(20) - cn10.shift(20))
C['wti_copper_rel_20d'] = (panel['WTI'].pct_change(20) - panel['COPPER'].pct_change(20))
C['wti_copper_rel_60d'] = (panel['WTI'].pct_change(60) - panel['COPPER'].pct_change(60))
C['oil_mom_20d'] = panel['WTI'].pct_change(20)
C['copper_mom_20d'] = panel['COPPER'].pct_change(20)
C['xau_mom_20d'] = panel['XAU'].pct_change(20)
C['btc_eth_rel_20d'] = (panel['BTC'].pct_change(20) - panel['ETH'].pct_change(20))
# --- autocorrelation / serial dependence ---
C['acorr_5d'] = ret.rolling(10).apply(lambda x: x.autocorr(1) if len(x) > 3 else np.nan, raw=False)
# --- composite / conditional ---
C['trend_x_vol'] = (panel.shift(5)/panel.shift(25)-1.0) * (vol20 / vol20.rolling(120).mean())
C['mom_20d_downside'] = (panel.shift(5)/panel.shift(25)-1.0) * (ret.clip(upper=0).rolling(20).std() / vol20)

def fast_ic(f, fr, min_assets=8):
    f = to_panel(f, fr.index)
    ics = []
    for d in fr.index:
        row = pd.concat([f.loc[d].rename('f'), fr.loc[d].rename('r')], axis=1).dropna()
        if len(row) >= min_assets:
            ic = row['f'].rank().corr(row['r'].rank())
            if np.isfinite(ic):
                ics.append(ic)
    return np.array(ics)

print(f"\n{'factor':<26}{'ic':>9}{'icir':>8}{'hit':>7}{'n':>6}  gate")
rows = []
for name, f in C.items():
    if f is None:
        continue
    ics = fast_ic(f, fwd)
    if len(ics) == 0:
        continue
    ic = float(ics.mean()); icir = ic / (float(ics.std(ddof=1)) or np.nan)
    hit = float((ics > 0).mean())
    flag = 'PASS' if (abs(ic) >= IC_TH and abs(icir) >= ICIR_TH) else ''
    print(f"{name:<26}{ic:>9.4f}{icir:>8.4f}{hit:>7.3f}{len(ics):>6d}  {flag}")
    rows.append((name, abs(ic), abs(icir), ic, icir, flag))

print(f"\n--- top 15 by |IC|*|ICIR| ---")
for name, aic, aicir, ic, icir, flag in sorted(rows, key=lambda r: r[1]*r[2], reverse=True)[:15]:
    print(f"{name:<26}{ic:>8.4f}{icir:>8.4f}  {flag}")
