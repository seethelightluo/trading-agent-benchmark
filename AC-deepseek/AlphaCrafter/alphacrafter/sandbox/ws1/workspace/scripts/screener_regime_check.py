"""Regime assessment for factor screener - uses data visible through 2026-07-15."""
import pandas as pd
import numpy as np
import glob, os

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"

def load(sym, folder=DATA):
    df = pd.read_csv(os.path.join(folder, sym + ".csv"), parse_dates=[0])
    df.columns = [c.strip() for c in df.columns]
    dcol = [c for c in df.columns if c.lower() in ("date", "datetime")][0]
    df = df.set_index(pd.to_datetime(df[dcol])).sort_index()
    ccol = [c for c in df.columns if c.lower() == "close"][0]
    return df[ccol]

panel = pd.DataFrame({s: load(s) for s in ASSETS})
idx_panel = pd.DataFrame({s: load(s, IDX) for s in ["VIX","DXY","USDJPY","EURUSD","USDCNY"]})
panel = panel[panel.index <= "2026-07-15"]
idx_panel = idx_panel[idx_panel.index <= "2026-07-15"]

print("Last date:", panel.index.max())
print("Panel shape:", panel.shape)

px = panel.dropna(how="all")
rets = px.pct_change()

# --- Trend metrics per asset ---
def trend_stats(s):
    x = px[s].dropna()
    if len(x) < 130:
        return None
    r5 = x.iloc[-1] / x.iloc[-6] - 1
    r20 = x.iloc[-1] / x.iloc[-21] - 1
    r60 = x.iloc[-1] / x.iloc[-61] - 1
    r120 = x.iloc[-1] / x.iloc[-121] - 1
    ma20 = x.rolling(20).mean().iloc[-1]
    ma60 = x.rolling(60).mean().iloc[-1]
    ma120 = x.rolling(120).mean().iloc[-1]
    price = x.iloc[-1]
    vol20 = rets[s].dropna().iloc[-20:].std() * np.sqrt(252)
    vol60 = rets[s].dropna().iloc[-60:].std() * np.sqrt(252)
    dd = (x / x.cummax() - 1).iloc[-1]
    return dict(r5=r5, r20=r20, r60=r60, r120=r120, price_vs_ma20=price/ma20-1,
                price_vs_ma60=price/ma60-1, price_vs_ma120=price/ma120-1,
                vol20=vol20, vol60=vol60, drawdown=dd)

rows = {}
for s in ASSETS:
    st = trend_stats(s)
    if st:
        rows[s] = st
tbl = pd.DataFrame(rows).T
pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
print("\n=== Per-asset trend/risk (through 2026-07-15) ===")
print(tbl.round(3).to_string())

print("\n=== Cross-asset breadth ===")
print("r20>0 count:", (tbl.r20 > 0).sum(), "/", len(tbl))
print("r60>0 count:", (tbl.r60 > 0).sum(), "/", len(tbl))
print("r120>0 count:", (tbl.r120 > 0).sum(), "/", len(tbl))
print("price>ma60 count:", (tbl.price_vs_ma60 > 0).sum(), "/", len(tbl))
print("price>ma120 count:", (tbl.price_vs_ma120 > 0).sum(), "/", len(tbl))
print("Median 20d vol (ann.):", tbl.vol20.median().round(3))
print("Mean 20d vol (ann.):", tbl.vol20.mean().round(3))
print("Median drawdown:", tbl.drawdown.median().round(3))

# --- Cross-sectional dispersion (for factor power) ---
cs_ret = rets.dropna(how="all")
disp20 = cs_ret.iloc[-20:].std(axis=1).mean()
disp60 = cs_ret.iloc[-60:].std(axis=1).mean()
print(f"\nCross-sectional daily dispersion last20d: {disp20:.4f}, last60d: {disp60:.4f}")

# --- Macro overlay ---
vix = idx_panel["VIX"].dropna()
dxy = idx_panel["DXY"].dropna()
print("\n=== Macro overlay ===")
print("VIX last:", round(vix.iloc[-1], 2), "| 20d ago:", round(vix.iloc[-21], 2),
      "| 60d ago:", round(vix.iloc[-61], 2), "| 60d min:", round(vix.iloc[-60:].min(), 2),
      "| 60d max:", round(vix.iloc[-60:].max(), 2))
print("VIX 20d change:", round(vix.iloc[-1]/vix.iloc[-21]-1, 4))
print("DXY last:", round(dxy.iloc[-1], 2), "| 60d ago:", round(dxy.iloc[-61], 2),
      "| 60d chg:", round(dxy.iloc[-1]/dxy.iloc[-61]-1, 4))

# --- Correlation regime (avg pairwise corr of 60d returns) ---
corr60 = rets.iloc[-60:].corr()
vals = corr60.values[np.triu_indices_from(corr60.values, k=1)]
vals = vals[np.isfinite(vals)]
print(f"\nAvg pairwise 60d return corr: {np.nanmean(vals):.3f}")

# --- Recent trend of the equal-weight cross-asset index ---
ew = px.mean(axis=1)
print("\n=== Equal-weight cross-asset basket ===")
for lbl, n in [("5d",5),("20d",20),("60d",60),("120d",120)]:
    print(f"{lbl} return: {ew.iloc[-1]/ew.iloc[-1-n]-1:.4f}")
print("EW price vs MA20:", round(ew.iloc[-1]/ew.rolling(20).mean().iloc[-1]-1, 4),
      "vs MA60:", round(ew.iloc[-1]/ew.rolling(60).mean().iloc[-1]-1, 4))
