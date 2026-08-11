"""miner_3 screen: quick horizon-10 IC screen for novel factor families (2026-08-27 cycle).
Candidates: fx_beta family (JPY/CNY/EUR), downside_beta_asym, amihud_illiq,
range_pos_60, tail_ratio_20, skew_term_structure, gap_ratio_20, xau_beta_60,
updown_vol_asym, zscore_60_price.
Only quick IC screen; deep validation happens per-candidate afterwards.
"""
import sys, math
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from miner3_lib import load_close_panel, rank_ic, WATCHLIST, load_ohlcv

C, V, H, L, O = load_close_panel(days=4000)
R = C.pct_change()

def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close']

DXY = load_macro('DXY'); VIX = load_macro('VIX')
USDJPY = load_macro('USDJPY'); USDCNY = load_macro('USDCNY'); EURUSD = load_macro('EURUSD')

# align macro to trading dates
def align_macro(s):
    s = s.reindex(C.index).ffill()
    return s

JPY = align_macro(USDJPY); CNY = align_macro(USDCNY); EUR = align_macro(EURUSD)

def rolling_beta(y, x, win):
    """rolling beta of y on x (both return series)"""
    yr, xr = y.pct_change(), x.pct_change()
    cov = yr.rolling(win).cov(xr)
    var = xr.rolling(win).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)

def fx_beta(xret, win=60):
    return {f'beta_{xret.name}_{win}': rolling_beta(C, xret, win) for xret in [JPY, CNY, EUR]}

def downside_beta_asym(win=60):
    """downside beta minus upside beta vs equal-weight cross-asset mean return"""
    mkt = R.mean(axis=1)
    out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
    for s in C.columns:
        y = R[s]
        beta_d, beta_u = [], []
        m = mkt.notna() & y.notna()
        mk, yy = mkt[m], y[m]
        for i in range(win, len(mk)):
            w = slice(i - win, i)
            mkw, yw = mk.iloc[w], yy.iloc[w]
            down = mkw < 0
            up = mkw > 0
            if down.sum() >= 5 and up.sum() >= 5:
                bd = np.cov(yw[down], mkw[down])[0, 1] / np.var(mkw[down])
                bu = np.cov(yw[up], mkw[up])[0, 1] / np.var(mkw[up])
                beta_d.append((mk.index[i], bd - bu))
        if beta_d:
            s_ = pd.Series(dict(beta_d))
            out.loc[s_.index, s] = s_.values
    return out

def amihud_illiq(win=20):
    il = (R.abs() / V.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return -il.rolling(win).mean()  # negated: higher = more liquid

def range_pos(win=60):
    hi = H.rolling(win).max()
    lo = L.rolling(win).min()
    rng = (hi - lo).replace(0, np.nan)
    return ((C - lo) / rng).replace([np.inf, -np.inf], np.nan)

def tail_ratio(win=20):
    """95th pct |ret| / 50th pct |ret| over window (fat-tail proxy)"""
    a = R.abs()
    p95 = a.rolling(win).quantile(0.95)
    p50 = a.rolling(win).quantile(0.50)
    return (p95 / p50.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def skew_term(short=10, long=60):
    sk_s = R.rolling(short).skew()
    sk_l = R.rolling(long).skew()
    return sk_s - sk_l

def gap_ratio(win=20):
    """avg |open-prev_close| / (high-low) -- overnight information share"""
    prev_close = C.shift(1)
    gap = (O - prev_close).abs()
    rng = (H - L).replace(0, np.nan)
    return (gap / rng).replace([np.inf, -np.inf], np.nan).rolling(win).mean()

def xau_beta(win=60):
    return rolling_beta(C, align_macro(C['XAU']), win)

def updown_vol_asym(win=20):
    """downside vol / upside vol asymmetry"""
    up = R.where(R > 0, np.nan)
    dn = R.where(R < 0, np.nan)
    vup = up.rolling(win).std()
    vdn = dn.rolling(win).std()
    return (vdn / vup.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def zscore_60():
    mu = C.rolling(60).mean()
    sd = C.rolling(60).std()
    return ((C - mu) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

cands = {}
cands.update(fx_beta(JPY)); cands.update(fx_beta(CNY)); cands.update(fx_beta(EUR))
cands['downside_beta_asym_60'] = downside_beta_asym()
cands['amihud_illiq_20'] = amihud_illiq()
cands['range_pos_60'] = range_pos()
cands['tail_ratio_20'] = tail_ratio()
cands['skew_term_10x60'] = skew_term()
cands['gap_ratio_20'] = gap_ratio()
cands['xau_beta_60'] = xau_beta()
cands['updown_vol_asym_20'] = updown_vol_asym()
cands['zscore_60'] = zscore_60()

FR = R.shift(-10)
print(f"{'factor':<24}{'IC':>8}{'ICIR':>8}{'hit':>7}{'n':>6}{'cov':>6}  regime(20-22/23-24/25-26)")
rows = []
for name, fp in cands.items():
    if fp is None or fp.shape[0] == 0:
        continue
    s = rank_ic(fp, FR)
    if s is None or len(s) < 30:
        print(f"{name:<24} insufficient dates ({0 if s is None else len(s)})")
        continue
    ic = s.mean(); icir = ic / s.std() if s.std() > 0 else 0.0
    hit = (s > 0).mean()
    cov = float(fp.notna().sum().sum()) / float(fp.size)
    regs = []
    for lo, hi in [("2020-01-01", "2022-12-31"), ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]:
        sub = s[(s.index >= lo) & (s.index <= hi)]
        regs.append(round(sub.mean(), 3) if len(sub) >= 20 else float('nan'))
    rows.append((name, ic, icir, hit, len(s), cov, regs))
    print(f"{name:<24}{ic:>8.4f}{icir:>8.4f}{hit:>7.3f}{len(s):>6}{cov:>6.2f}  {regs[0]}/{regs[1]}/{regs[2]}")

print("\n--- Top by |IC|*|ICIR| ---")
rows.sort(key=lambda r: abs(r[1] * r[2]), reverse=True)
for r in rows[:12]:
    print(f"{r[0]:<24} IC={r[1]:.4f} ICIR={r[2]:.4f} hit={r[3]:.3f} n={r[4]} cov={r[5]:.2f} regime={r[6]}")
