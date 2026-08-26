"""miner_3 exploration: gold (XAU) beta safe-haven sensitivity factor.</think>
Fully scripted below.
"""
import pandas as pd, numpy as np

TICKERS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
           "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
HORIZON = 10
WINDOW = 60
END = pd.Timestamp("2033-06-22")

# load close/ret
close = {}
ret = {}
for t in TICKERS:
    df = pd.read_csv(f"../persistent/stock_data/{t}.csv", parse_dates=["date"])
    df = df[df["date"] <= END].sort_values("date").set_index("date")
    r = df["close"].pct_change()
    close[t] = df["close"]
    ret[t] = r

retdf = pd.DataFrame(ret)
closef = pd.DataFrame(close)

xau = retdf["XAU"]

def factor_beta(asset_ret, bench_ret, window):
    # factor value at date t uses returns through t (shift applied to covariance)
    cov = (asset_ret * bench_ret).rolling(window).mean()
    var = (bench_ret * bench_ret).rolling(window).mean()
    f = cov / var
    return f.shift(1)  # use only info available at start of holding period

fdf = pd.DataFrame({t: factor_beta(retdf[t], xau, WINDOW) for t in TICKERS})

# Forward returns
fwd = closef.shift(-HORIZON) / closef - 1

# align
f = fdf.reindex(fwd.index)
fr = fwd.reindex(fwd.index)

ics = []
dates = []
for dt in f.index:
    x = f.loc[dt].dropna()
    y = fr.loc[dt].reindex(x.index).dropna()
    common = x.index.intersection(y.index)
    if len(common) < 8:
        continue
    xa = x[common].values
    ya = y[common].values
    if np.std(xa)==0 or np.std(ya)==0:
        continue
    ic = np.corrcoef(xa, ya)[0,1]
    if not np.isnan(ic):
        ics.append(ic)
        dates.append(dt)

ics = np.array(ics)
print("GOLD-XAU-BETA factor, horizon", HORIZON, "window", WINDOW)
print("n_ic_dates:", len(ics))
print("dates range:", dates[0], "->", dates[-1])
print("mean IC:", round(ics.mean(),4))
print("ICIR:", round(ics.mean()/ics.std(),4))
print("IC hit ratio (abs>0):", round((ics>0).mean(),4))
print("IC std:", round(ics.std(),4))
print("coverage_asset_days:", round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),4))
# signs: is higher gold beta predictive? we report raw sign