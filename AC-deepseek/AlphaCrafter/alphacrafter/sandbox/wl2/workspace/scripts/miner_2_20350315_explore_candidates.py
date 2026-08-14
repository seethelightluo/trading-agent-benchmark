"""miner_2 exploration 2035-03-15: screen novel factor candidates on the 15-asset benchmark.
Data through visible_through=2035-03-14 only (no future leakage).
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

SYMS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END = pd.Timestamp('2035-03-14')

def load_col(colname):
    cols = {}
    for s in SYMS:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        cols[s] = df.set_index('date')[colname]
    out = pd.DataFrame(cols).sort_index()
    out = out[out.index <= END]
    return out

P = load_col('close')
O = load_col('open')
H = load_col('high')
L = load_col('low')
R = P.pct_change()
VIX = pd.read_csv('../persistent/index_data/VIX.csv')
VIX['date'] = pd.to_datetime(VIX['date']); VIX = VIX.set_index('date')['close']; VIX = VIX[VIX.index <= END]

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def roll_vol(x, n):
    return x.rolling(n).std(ddof=0)

F = {}
# 1 autocorr_20
F['autocorr_20'] = R.rolling(20).apply(lambda x: pd.Series(x).autocorr() if x.std() > 0 else np.nan, raw=False)
# 2 var_ratio_5_60: VR = var(5d rets)/(5*var(1d rets)) over 60d
r5 = R.rolling(5).sum()
F['var_ratio_5_60'] = (r5.rolling(60).var(ddof=0)) / (5.0 * R.rolling(60).var(ddof=0))
# 3 updown_capture_60
up = R.where(R > 0, np.nan).rolling(60).mean()
dn = R.where(R < 0, np.nan).rolling(60).mean()
F['updown_capture_60'] = up / dn.abs()
# 4 crash_recovery_5_20: (r5 - r20)/vol20
r20 = R.rolling(20).sum()
F['crash_recovery_5_20'] = (R.rolling(5).sum() - r20) / roll_vol(R, 20)
# 5 mom_sharpe_60_20
F['mom_sharpe_60_20'] = R.rolling(60).sum() / roll_vol(R, 20)
# 6 ema_cross_10_40
F['ema_cross_10_40'] = (ema(P, 10) - ema(P, 40)) / (roll_vol(R, 10) * P)
# 7 drawdown_depth_60
F['drawdown_depth_60'] = (P - P.rolling(60).max()) / P.rolling(60).max()
# 8 skew_60
F['skew_60'] = R.rolling(60).skew()
# 9 tail_ratio_60
F['tail_ratio_60'] = R.rolling(60).quantile(0.95) / R.rolling(60).quantile(0.05).abs()
# 10 xau_corr_60
F['xau_corr_60'] = R.rolling(60).corr(R['XAU'])
# 11 rsi_14
delta = R
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
F['rsi_14'] = 100 - 100 / (1 + rs)
# 12 gap_mean_20
gap = O / P.shift(1) - 1
F['gap_mean_20'] = gap.rolling(20).mean() / roll_vol(R, 20)
# 13 breakout_freq_60: fraction of days close > prior 20d max
prior_hi = H.rolling(20).max().shift(1)
F['breakout_freq_60'] = (P > prior_hi).rolling(60).mean()
# 14 vol_spread_5_60
F['vol_spread_5_60'] = roll_vol(R, 5) / roll_vol(R, 60) - 1
# 15 dist_ma50_vol
F['dist_ma50_vol'] = (P - P.rolling(50).mean()) / (roll_vol(R, 20) * P)
# 16 upday_fraction_60
F['upday_fraction_60'] = (R > 0).rolling(60).mean()
# 17 hi_lo_range_20: avg daily range / vol
F['hi_lo_range_20'] = ((H - L) / P).rolling(20).mean() / roll_vol(R, 20)

def ic_series(fv, fwd, min_valid=8):
    out = {}
    for t in fv.index:
        x = fv.loc[t]
        y = fwd.loc[t]
        m = x.notna() & y.notna()
        if m.sum() < min_valid:
            continue
        rho, _ = spearmanr(x[m], y[m])
        out[t] = rho
    return pd.Series(out)

def summarize(name, fv, fwd, start=pd.Timestamp('2021-01-01')):
    row = {}
    for h, ff in fwd.items():
        ic = ic_series(fv, ff)
        sub = ic[ic.index >= start]
        if len(sub) == 0:
            row[f'ic{h}'] = np.nan; row[f'icir{h}'] = np.nan; row[f'n{h}'] = 0
            continue
        row[f'ic{h}'] = sub.mean()
        row[f'icir{h}'] = sub.mean() / sub.std() if sub.std() > 0 else np.nan
        row[f'n{h}'] = len(sub)
        row[f'hit{h}'] = (sub > 0).mean()
    return row

fwd = {5: P.shift(-5) / P - 1, 10: P.shift(-10) / P - 1}
rows = []
for name, fv in F.items():
    r = summarize(name, fv, fwd)
    r['factor'] = name
    rows.append(r)
res = pd.DataFrame(rows).set_index('factor')
print('=== Candidate screen (2021-01-01 .. 2035-03-14), horizon 5 and 10 ===')
print(res.round(4).to_string())
print()
print('=== Pass gate (|ic|>=0.007 & |icir|>=0.084), both horizons checked ===')
for name, r in res.iterrows():
    for h in (5, 10):
        if abs(r[f'ic{h}']) >= 0.007 and abs(r[f'icir{h}']) >= 0.084 and r[f'n{h}'] > 500:
            print(f'{name:24s} h{h}: ic={r[f"ic{h}"]:.4f} icir={r[f"icir{h}"]:.4f} hit={r[f"hit{h}"]:.3f} n={r[f"n{h}"]}')
