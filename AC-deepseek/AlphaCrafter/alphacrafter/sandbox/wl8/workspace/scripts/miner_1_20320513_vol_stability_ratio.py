"""
Factor: vol_stability_ratio_20x60
Idea: Ratio of recent 20-day volatility to longer-term 60-day volatility.
Direction: SHORT (lower ratio = more stable = higher rank)
Gate: abs(IC) >= 0.0070, abs(ICIR) >= 0.0840
"""

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import numpy as np
from scipy.stats import spearmanr

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
         'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get_close(sym, days=400):
    df = get_stock_daily_data(sym, days=days)
    if df is not None and 'close' in df.columns and len(df) > 70:
        return np.array(df['close'].values, dtype=float), np.array(df['date'].values, dtype='datetime64[D]')
    df = get_index_daily_data(sym, days=days)
    if df is not None and 'close' in df.columns and len(df) > 70:
        return np.array(df['close'].values, dtype=float), np.array(df['date'].values, dtype='datetime64[D]')
    return None, None

print("Loading data...")
all_closes, all_dates = {}, {}
for sym in WATCH:
    c, d = get_close(sym)
    if c is not None:
        all_closes[sym] = c
        all_dates[sym] = d
usable = [s for s in WATCH if s in all_closes]
print(f"Usable assets: {len(usable)}")

# 5-day forward horizon
factor_vals_5d, fwd_ret_5d = {}, {}
for sym in usable:
    c = all_closes[sym]; d = all_dates[sym]
    rets = np.diff(c) / c[:-1]
    for t in range(65, len(rets)-5):
        dkey = str(d[t])
        if dkey not in factor_vals_5d:
            factor_vals_5d[dkey] = {}; fwd_ret_5d[dkey] = {}
        v20 = float(np.std(rets[t-20:t]))
        v60 = float(np.std(rets[t-60:t]))
        if v60 < 1e-10: continue
        factor_vals_5d[dkey][sym] = v20 / v60
        fwd_ret_5d[dkey][sym] = c[t+5]/c[t]-1.0 if t+5<len(c) else c[-1]/c[t]-1.0

ics5 = []
for dkey in sorted(factor_vals_5d.keys()):
    fv = factor_vals_5d[dkey]; fr = fwd_ret_5d[dkey]
    com = [s for s in fv if s in fr and np.isfinite(fv[s]) and np.isfinite(fr[s])]
    if len(com) >= 8:
        rho, _ = spearmanr([fv[s] for s in com], [fr[s] for s in com])
        ics5.append(rho)

ics5 = np.array(ics5)
ic5_m = float(np.mean(ics5)); ic5_s = float(np.std(ics5)); ic5_ir = ic5_m/ic5_s if ic5_s>0 else 0
hit5 = float(np.sum(ics5>0))/len(ics5) if len(ics5)>0 else 0
print(f"\n=== VOL STABILITY RATIO 20x60 (SHORT) - 5d forward ===")
print(f"Dates: {len(ics5)}")
print(f"IC mean: {ic5_m:.6f}  ICIR: {ic5_ir:.6f}  Hit: {hit5:.3f}")
print(f"Gate IC>=0.007: {'PASS' if abs(ic5_m)>=0.007 else 'FAIL'}")
print(f"Gate ICIR>=0.084: {'PASS' if abs(ic5_ir)>=0.084 else 'FAIL'}")

# 10-day forward horizon
factor_vals_10d, fwd_ret_10d = {}, {}
for sym in usable:
    c = all_closes[sym]; d = all_dates[sym]
    rets = np.diff(c) / c[:-1]
    for t in range(65, len(rets)-10):
        dkey = str(d[t])
        if dkey not in factor_vals_10d:
            factor_vals_10d[dkey] = {}; fwd_ret_10d[dkey] = {}
        v20 = float(np.std(rets[t-20:t]))
        v60 = float(np.std(rets[t-60:t]))
        if v60 < 1e-10: continue
        factor_vals_10d[dkey][sym] = v20 / v60
        fwd_ret_10d[dkey][sym] = c[t+10]/c[t]-1.0 if t+10<len(c) else c[-1]/c[t]-1.0

ics10 = []
for dkey in sorted(factor_vals_10d.keys()):
    fv = factor_vals_10d[dkey]; fr = fwd_ret_10d[dkey]
    com = [s for s in fv if s in fr and np.isfinite(fv[s]) and np.isfinite(fr[s])]
    if len(com) >= 8:
        rho, _ = spearmanr([fv[s] for s in com], [fr[s] for s in com])
        ics10.append(rho)

ics10 = np.array(ics10)
ic10_m = float(np.mean(ics10)); ic10_s = float(np.std(ics10)); ic10_ir = ic10_m/ic10_s if ic10_s>0 else 0
hit10 = float(np.sum(ics10>0))/len(ics10) if len(ics10)>0 else 0
print(f"\n=== VOL STABILITY RATIO 20x60 (SHORT) - 10d forward ===")
print(f"Dates: {len(ics10)}")
print(f"IC mean: {ic10_m:.6f}  ICIR: {ic10_ir:.6f}  Hit: {hit10:.3f}")
print(f"Gate IC>=0.007: {'PASS' if abs(ic10_m)>=0.007 else 'FAIL'}")
print(f"Gate ICIR>=0.084: {'PASS' if abs(ic10_ir)>=0.084 else 'FAIL'}")

# Rank turnover
print("\n=== Turnover (rank delta > 0.25 fraction) ===")
dates_s = sorted(factor_vals_10d.keys())
turns = []
for i in range(1, len(dates_s)):
    fv1 = factor_vals_10d[dates_s[i-1]]
    fv2 = factor_vals_10d[dates_s[i]]
    com = [s for s in fv1 if s in fv2 and np.isfinite(fv1[s]) and np.isfinite(fv2[s])]
    if len(com) >= 8:
        r1_list = sorted([(fv1[s],s) for s in com])
        r2_list = sorted([(fv2[s],s) for s in com])
        r1 = {s: i/max(1,len(com)-1) for i,(_,s) in enumerate(r1_list)}
        r2 = {s: i/max(1,len(com)-1) for i,(_,s) in enumerate(r2_list)}
        big = sum(1 for s in com if abs(r1[s]-r2[s]) > 0.25)
        turns.append(big/len(com))
if turns:
    print(f"Mean turnover: {np.mean(turns):.3f}")

# Current factor values
print("\n=== Current factor values (2032-05-12) ===")
now_vals = {}
for sym in usable:
    c = all_closes[sym]; rets = np.diff(c)/c[:-1]
    if len(rets) >= 65:
        v20 = float(np.std(rets[-20:]))
        v60 = float(np.std(rets[-60:]))
        if v60 >= 1e-10:
            now_vals[sym] = v20/v60
sorted_vals = sorted(now_vals.items(), key=lambda x: x[1])
for sym, v in sorted_vals:
    print(f"  {sym}: {v:.4