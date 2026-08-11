"""Regime assessment for factor screener cycle - data visible through 2026-07-29 only."""
import pandas as pd
import numpy as np

VISIBLE = "2026-07-29"
UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OBS = {"DXY": "../persistent/index_data/DXY.csv",
       "VIX": "../persistent/index_data/VIX.csv",
       "USDJPY": "../persistent/index_data/USDJPY.csv",
       "USDCNY": "../persistent/index_data/USDCNY.csv",
       "EURUSD": "../persistent/index_data/EURUSD.csv"}

def load(path, name):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

prices = {}
for s in UNIVERSE:
    d = load(f"../persistent/stock_data/{s}.csv", s)
    prices[s] = d["close"]
px = pd.DataFrame(prices).dropna(how="all")
rets = px.pct_change()

print("=" * 100)
print(f"REGIME ASSESSMENT (visible through {VISIBLE}) | n_days={len(px)}")
print("=" * 100)

# --- Trend: returns over horizons ---
print("\n[1] TREND - cumulative returns by horizon (%)")
hors = [5, 21, 63, 126, 252]
last = px.iloc[-1]
for h in hors:
    if len(px) > h:
        base = px.iloc[-1 - h]
        cum = (last / base - 1) * 100
        print(f"  h={h:>4}: " + "  ".join(f"{s}:{cum[s]:+7.1f}" for s in UNIVERSE))
    else:
        print(f"  h={h:>4}: insufficient")

# EW market proxy
ew = px.mean(axis=1)
ew_ret = ew.pct_change()
for h in hors:
    if len(px) > h:
        print(f"  EW-basket h={h:>4}: {(ew.iloc[-1]/ew.iloc[-1-h]-1)*100:+.2f}%")

# --- Trend strength: 20/60 SMA slope, consecutive up days ---
print("\n[2] TREND STRENGTH")
for s in UNIVERSE:
    c = px[s].dropna()
    if len(c) < 70:
        continue
    sma20 = c.rolling(20).mean().iloc[-1]
    sma60 = c.rolling(60).mean().iloc[-1]
    slope20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) > 21 else np.nan
    r = c.pct_change().dropna()
    consec_up = 0
    for v in r.iloc[::-1]:
        if v > 0:
            consec_up += 1
        else:
            break
    regime = "UP" if c.iloc[-1] > sma60 else "DOWN"
    print(f"  {s:<10} close={c.iloc[-1]:>10.2f} sma20={sma20:>10.2f} sma60={sma60:>10.2f} "
          f"20d_slope={slope20:+6.2f}% consec_up={consec_up:>2} regime_vs60={regime}")

# --- Risk: realized vol ---
print("\n[3] RISK - realized vol (ann. %)")
for s in UNIVERSE:
    r = rets[s].dropna()
    if len(r) > 20:
        v20 = r.iloc[-20:].std() * np.sqrt(252) * 100
        v60 = r.iloc[-60:].std() * np.sqrt(252) * 100 if len(r) > 60 else np.nan
        print(f"  {s:<10} vol20={v20:6.1f}%  vol60={v60:6.1f}%")

# --- Correlation regime ---
print("\n[4] CORRELATION REGIME - avg pairwise abs corr (60d)")
r60 = rets.iloc[-60:]
corr = r60.corr()
mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
avg_abs = corr.where(mask).abs().stack().mean()
avg_pos = corr.where(mask).stack().mean()
print(f"  60d avg |corr| = {avg_abs:.3f}, avg corr = {avg_pos:.3f} (n_assets={corr.shape[0]})")
# dispersion
disp = r60.std(axis=1).mean() * np.sqrt(252) * 100
print(f"  60d cross-sectional dispersion (ann.) = {disp:.2f}%")

# --- Volatility regime (EW basket) ---
print("\n[5] VOLATILITY REGIME (EW basket)")
ewv20 = ew_ret.iloc[-20:].std() * np.sqrt(252) * 100
ewv60 = ew_ret.iloc[-60:].std() * np.sqrt(252) * 100
ewv252 = ew_ret.iloc[-252:].std() * np.sqrt(252) * 100 if len(ew_ret) > 252 else np.nan
print(f"  EW-basket vol20={ewv20:.2f}% vol60={ewv60:.2f}% vol252={ewv252:.2f}%")

# --- Observation signals ---
print("\n[6] MACRO OBSERVATION SIGNALS")
for name, path in OBS.items():
    d = load(path, name)
    c = d["close"]
    c20 = c.iloc[-21] if len(c) > 21 else c.iloc[0]
    c60 = c.iloc[-61] if len(c) > 61 else c.iloc[0]
    print(f"  {name:<7} last={c.iloc[-1]:>10.4f}  20d_chg={(c.iloc[-1]/c20-1)*100:+6.2f}%  60d_chg={(c.iloc[-1]/c60-1)*100:+6.2f}%")

# --- Recent drawdown from 252d high ---
print("\n[7] DRAWDOWN from 1y high (%)")
for s in UNIVERSE:
    c = px[s].dropna()
    if len(c) < 252:
        continue
    hh = c.iloc[-252:].max()
    dd = (c.iloc[-1] / hh - 1) * 100
    print(f"  {s:<10} dd_from_1y_high={dd:7.2f}%")

print("\nDone.")
