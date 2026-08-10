"""miner_2 broad screen v2 2026-07-30: relaxed min_periods for rolling windows.

Rationale: panel reindexed to macro calendar has ~1-11% scattered holiday NaNs;
rolling windows with default min_periods=window lose most dates for 60d stats.
Using min_periods = win//2 gives stable estimates with far better coverage.
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
print(f"panel: {len(panel)} dates x {panel.shape[1]} assets, "
      f"{panel.index.min().date()} .. {panel.index.max().date()}")

def roll_std(x, w):
    return x.rolling(w, min_periods=max(10, w // 2)).std()

def roll_mean(x, w):
    return x.rolling(w, min_periods=max(10, w // 2)).mean()

def roll_skew(x, w):
    return x.rolling(w, min_periods=max(10, w // 2)).skew()

def roll_kurt(x, w):
    return x.rolling(w, min_periods=max(10, w // 2)).kurt()

dxy = macro['DXY']; vix = macro['VIX']; jpy = macro['USDJPY']
cny = macro['USDCNY']; eur = macro['EURUSD']
dxy_r = dxy.pct_change(); vix_r = vix.pct_change()
jpy_r = jpy.pct_change(); cny_r = cny.pct_change(); eur_r = eur.pct_change()

def beta_of(a, m, win):
    return a.rolling(win, min_periods=max(15, win // 2)).cov(m) / m.rolling(win, min_periods=max(15, win // 2)).var()

def to_panel(f, idx):
    if isinstance(f, pd.Series):
        f = f.to_frame(name=f.name or 'f0')
    return f.reindex(idx)

C = {}
vol20 = roll_std(ret, 20); vol60 = roll_std(ret, 60); vol10 = roll_std(ret, 10)
C['vol_ratio_10_60'] = vol10 / vol60
C['vol_ratio_20_60'] = vol20 / vol60
C['vol_zscore_20_120'] = (vol20 - roll_mean(vol20, 120)) / roll_std(vol20, 120)
C['downside_vol_20d'] = -ret.clip(upper=0).rolling(20, min_periods=10).std()
C['downside_ratio_20_60'] = ret.clip(upper=0).rolling(20, min_periods=10).std() / ret.clip(upper=0).rolling(60, min_periods=30).std()
C['range_20d'] = (panel.rolling(20, min_periods=10).max() - panel.rolling(20, min_periods=10).min()) / panel
C['range_60d'] = (panel.rolling(60, min_periods=30).max() - panel.rolling(60, min_periods=30).min()) / panel
C['parkinson_20d'] = np.log(panel.rolling(20, min_periods=10).max()/panel.rolling(20, min_periods=10).min()) / (2*np.sqrt(2*np.log(2)))
C['skew_60d'] = roll_skew(ret, 60)
C['kurt_60d'] = roll_kurt(ret, 60)
C['skew_20d'] = roll_skew(ret, 20)
C['amihud_20d'] = (ret.abs() / panel).rolling(20, min_periods=10).mean()
C['amihud_60d'] = (ret.abs() / panel).rolling(60, min_periods=30).mean()
C['risk_adj_mom_60d_skip5'] = (panel.shift(5) / panel.shift(65) - 1.0) / vol60
C['risk_adj_mom_20d_skip5'] = (panel.shift(5) / panel.shift(25) - 1.0) / vol20
C['sharpish_60d'] = roll_mean(ret, 60) / vol60
C['dxy_beta_60d'] = beta_of(ret, dxy_r, 60)
C['dxy_beta_120d'] = beta_of(ret, dxy_r, 120)
C['jpy_beta_60d'] = beta_of(ret, jpy_r, 60)
C['eur_beta_60d'] = beta_of(ret, eur_r, 60)
C['cny_beta_60d'] = beta_of(ret, cny_r, 60)
C['vix_beta_60d'] = beta_of(ret, vix_r, 60)
C['trend_x_vol'] = (panel.shift(5)/panel.shift(25)-1.0) * (vol20 / roll_mean(vol20, 120))
C['mom_20d_downside'] = (panel.shift(5)/panel.shift(25)-1.0) * (ret.clip(upper=0).rolling(20, min_periods=10).std() / vol20)
C['mdd_60d'] = panel / panel.rolling(60, min_periods=30).max() - 1.0
C['dist_52w_high'] = panel / panel.rolling(250, min_periods=125).max() - 1.0
C['dist_52w_low'] = panel / panel.rolling(250, min_periods=125).min() - 1.0

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

print(f"\n{'factor':<26}{'ic':>9}{'icir':>8}{'hit':>7}{'n':>6}  gate  ic_date_range")
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
    lo, hi = None, None
    # need dates: redo quickly with dates
    print(f"{name:<26}{ic:>9.4f}{icir:>8.4f}{hit:>7.3f}{len(ics):>6d}  {flag}")
    rows.append((name, abs(ic), abs(icir), ic, icir, flag, len(ics)))

print(f"\n--- top 18 by |IC|*|ICIR| (with n_dates) ---")
for name, aic, aicir, ic, icir, flag, n in sorted(rows, key=lambda r: r[1]*r[2], reverse=True)[:18]:
    print(f"{name:<26}{ic:>8.4f}{icir:>8.4f}  n={n:<5} {flag}")
