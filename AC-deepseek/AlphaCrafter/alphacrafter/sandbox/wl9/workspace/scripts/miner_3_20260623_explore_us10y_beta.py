"""miner_3 exploration: US10Y rate-sensitivity beta factor.

Cross-sectional beta of each asset's daily return to US10Y yield return.
Direction: assets that appreciate with falling yields (defensives) vs assets
that appreciate with rising yields. Test predictive power over forward 10d.
"""
import pandas as pd, numpy as np

TICKERS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
           "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
HORIZON = 10
WINDOW = 60
END = pd.Timestamp("2033-06-22")

close, ret = {}, {}
for t in TICKERS:
    df = pd.read_csv(f"../persistent/stock_data/{t}.csv", parse_dates=["date"])
    df = df[df["date"] <= END].sort_values("date").set_index("date")
    r = df["close"].pct_change()
    close[t] = df["close"]
    ret[t] = r
retdf = pd.DataFrame(ret)
closef = pd.DataFrame(close)
bench = retdf["US10Y"]

def factor_beta(asset_ret, bench_ret, window):
    cov = (asset_ret*bench_ret).rolling(window).mean()
    var = (bench_ret*bench_ret).rolling(window).mean()
    return (cov/var).shift(1)

fdf = pd.DataFrame({t: factor_beta(retdf[t], bench, WINDOW) for t in TICKERS})
fwd = closef.shift(-HORIZON)/closef - 1

ics, dates = [], []
for dt in fdf.index:
    x = fdf.loc[dt].dropna()
    y = fwd.loc[dt].dropna()
    common = x.index.intersection(y.index)
    if len(common) < 8: continue
    xa, ya = x[common].values, y[common].values
    if np.std(xa)==0 or np.std(ya)==0: continue
    ic = np.corrcoef(xa, ya)[0,1]
    if not np.isnan(ic):
        ics.append(ic); dates.append(dt)
ics = np.array(ics)
print("US10Y-BETA factor h", HORIZON, "window", WINDOW)
print("n_ic_dates:", len(ics), "range:", dates[0].date(), "->", dates[-1].date())
print("mean IC:", round(ics.mean(),4), "ICIR:", round(ics.mean()/ics.std(),4),
      "IC_std:", round(ics.std(),4))
print("hit_ratio(>0):", round((ics>0).mean(),4))
print("coverage:", round(fdf.notna().sum().sum()/(fdf.shape[0]*fdf.shape[1]),4))