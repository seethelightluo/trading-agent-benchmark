"""Screener regime & factor assessment as of 2030-02-11 (data through visible_through 2030-02-08)."""
import pandas as pd
import numpy as np

TRADABLE = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
OBS = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
END = "2030-02-08"

px = {}
for s in TRADABLE + OBS:
    df = pd.read_csv(f"../persistent/{("index_data" if s in OBS else "stock_data")}/{s}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END].set_index("date").sort_index()
    px[s] = df["close"]

PX = pd.DataFrame(px)
ret = PX.pct_change()

print("=== Last close (2030-02-08) ===")
print(PX.tail(1).T.round(2).to_string())

print("\n=== Recent returns by asset ===")
out = pd.DataFrame({
    "r5d": PX.pct_change(5).iloc[-1],
    "r10d": PX.pct_change(10).iloc[-1],
    "r20d": PX.pct_change(20).iloc[-1],
    "r60d": PX.pct_change(60).iloc[-1],
    "vol20": ret.rolling(20).std().iloc[-1],
})
out = out.reindex(TRADABLE)
print(out.round(4).to_string())

print("\n=== Frozen/stale check (last 10 trading days pct_change == 0) ===")
for s in TRADABLE:
    z = (ret[s].tail(10) == 0).sum()
    if z >= 5:
        print(f"  {s}: {z}/10 zero-return days")

print("\n=== Regime metrics ===")
print("VIX last:", round(PX['VIX'].iloc[-1],1), "| VIX 20d mean:", round(PX['VIX'].tail(20).mean(),1),
      "| VIX 60d mean:", round(PX['VIX'].tail(60).mean(),1), "| VIX 20d ago:", round(PX['VIX'].iloc[-21],1))
# trend: SPX vs 60d MA
for s in ["SPX","NDX","000300.SH","N225","SX5E","HSI","000688.SH","XAU","COPPER","WTI","BTC","US10Y","CN10Y"]:
    c = PX[s].iloc[-1]; ma60 = PX[s].tail(60).mean(); ma20 = PX[s].tail(20).mean()
    print(f"  {s}: last={c:.1f} ma20={ma20:.1f} ma60={ma60:.1f} vs_ma60={(c/ma60-1)*100:+.1f}%")
# avg pairwise correlation of 15 tradable (last 60d)
c = ret[TRADABLE].tail(60).corr()
vals = c.values[np.triu_indices(len(TRADABLE), k=1)]
print("avg pairwise corr 60d (15 assets):", round(np.nanmean(vals),3))
c20 = ret[TRADABLE].tail(20).corr()
vals20 = c20.values[np.triu_indices(len(TRADABLE), k=1)]
print("avg pairwise corr 20d (15 assets):", round(np.nanmean(vals20),3))

# ---- Factor values as of END ----
print("\n=== Factor values as of 2030-02-08 ===")
mkt = ret[TRADABLE].mean(axis=1)
down = mkt.where(mkt < 0)
res = {}
# dn_mkt_beta_60d
b = {}
for s in TRADABLE:
    d = pd.concat([ret[s], down], axis=1).dropna()
    d = d.tail(60)
    if len(d) >= 40 and d.iloc[:,1].std() > 0:
        cov = np.cov(d.iloc[:,0], d.iloc[:,1])[0,1]
        b[s] = cov / d.iloc[:,1].var()
    else:
        b[s] = np.nan
res["dn_mkt_beta_60d"] = pd.Series(b)
# rate_beta_cn10y_60d
rb = {}
cn = ret["CN10Y"]
for s in TRADABLE:
    d = pd.concat([ret[s], cn], axis=1).dropna().tail(60)
    if len(d) >= 40 and d.iloc[:,1].std() > 1e-12:
        cov = np.cov(d.iloc[:,0], d.iloc[:,1])[0,1]
        rb[s] = cov / d.iloc[:,1].var()
    else:
        rb[s] = np.nan
res["rate_beta_cn10y_60d"] = pd.Series(rb)
# vol_adj_mom_accel_20x60
va = {}
for s in TRADABLE:
    c = PX[s]
    if len(c) < 80:
        va[s] = np.nan; continue
    mom20 = c.iloc[-1]/c.iloc[-21] - 1
    mom60 = c.iloc[-1]/c.iloc[-61] - 1
    sd = ret[s].tail(20).std()
    va[s] = (mom20 - mom60)/sd if sd > 0 else np.nan
res["vol_adj_mom_accel_20x60"] = pd.Series(va)

F = pd.DataFrame(res)
print(F.round(4).to_string())

print("\n=== Cross-sectional rank (1=lowest factor value) ===")
print(F.rank().round(2).to_string())

# factor correlation using daily factor series (approximate, 60d)
print("\n=== Factor cross-correlation (recent 60d daily series) ===")
fs = {}
mkt_r = ret[TRADABLE].mean(axis=1)
for s in TRADABLE:
    pass
# build daily factor series
dseries = {}
for fname in ["dn_mkt_beta_60d","rate_beta_cn10y_60d"]:
    ser = {}
    for s in TRADABLE:
        if fname == "dn_mkt_beta_60d":
            d = pd.concat([ret[s], down], axis=1).dropna()
            d = d.tail(60)
            if len(d) >= 40 and d.iloc[:,1].std() > 0:
                ser[s] = np.cov(d.iloc[:,0], d.iloc[:,1])[0,1]/d.iloc[:,1].var()
            else:
                ser[s] = np.nan
        else:
            d = pd.concat([ret[s], cn], axis=1).dropna().tail(60)
            if len(d) >= 40 and d.iloc[:,1].std() > 1e-12:
                ser[s] = np.cov(d.iloc[:,0], d.iloc[:,1])[0,1]/d.iloc[:,1].var()
            else:
                ser[s] = np.nan
    dseries[fname] = pd.Series(ser)
dseries["vol_adj_mom_accel_20x60"] = va
D = pd.DataFrame(dseries)
print(D.corr().round(3).to_string())

print("\n=== Forward 10d returns (last observed window, for feedback) ===")
# realized forward 10d returns for the most recent completed 10d blocks
fwd = {}
for s in TRADABLE:
    fwd[s] = PX[s].shift(-10)/PX[s] - 1
FWD = pd.DataFrame(fwd).iloc[-10:]
print(FWD.round(4).to_string())
