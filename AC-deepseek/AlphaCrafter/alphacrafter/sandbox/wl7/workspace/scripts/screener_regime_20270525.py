"""Screener regime snapshot for cycle 2027-05-25 (data through 2027-05-24).
Reads persistent CSVs but truncates at visible date to avoid lookahead."""
import pandas as pd
import numpy as np

VISIBLE = "2027-05-24"
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

def load(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].reset_index(drop=True)
    return df

px = {}
for a in ASSETS:
    df = load(f"../persistent/stock_data/{a}.csv")
    px[a] = df.set_index("date")["close"]
px = pd.DataFrame(px).dropna(how="all").sort_index()

macro = {}
for m in MACRO:
    df = load(f"../persistent/index_data/{m}.csv")
    macro[m] = df.set_index("date")["close"]
macro = pd.DataFrame(macro).dropna(how="all").sort_index()

ret = px.pct_change()
last = px.iloc[-1]
r20 = (px.iloc[-1] / px.iloc[-21] - 1) * 100 if len(px) > 21 else np.nan
r60 = (px.iloc[-1] / px.iloc[-61] - 1) * 100 if len(px) > 61 else np.nan
ma20 = px.rolling(20).mean().iloc[-1]
c_ma20 = last / ma20
vol20 = ret.tail(20).std() * np.sqrt(252) * 100
down_ret = ret[ret < 0]
dvol20 = down_ret.tail(20).std() * np.sqrt(252) * 100

print("=== CROSS-SECTION SNAPSHOT @", VISIBLE, "===")
out = pd.DataFrame({
    "close": last, "ret20d%": r20, "ret60d%": r60, "c/MA20": c_ma20,
    "vol20_ann%": vol20, "dvol20_ann%": dvol20})
print(out.round(3).to_string())

print("\n=== MACRO ===")
print(macro.tail(1).round(3).to_string())
mv20 = (macro.iloc[-1] / macro.iloc[-21] - 1) * 100
print("macro 20d %:", mv20.round(2).to_dict())

# VIX path
vix = macro["VIX"]
print("\nVIX last 10 closes:", vix.tail(10).round(2).tolist())
print("VIX 20d ago:", round(vix.iloc[-21], 2) if len(vix) > 21 else None)
print("VIX 5d ago:", round(vix.iloc[-6], 2))

# Correlation regime: 20d vs 60d mean abs pairwise corr on returns (live 15 assets)
c20 = ret.tail(20).corr().abs().values
c60 = ret.tail(60).corr().abs().values
c120 = ret.tail(120).corr().abs().values
n = c20.shape[0]
mask = ~np.eye(n, dtype=bool)
print("\nmean |corr| 20d:", round(c20[mask].mean(), 4),
      " 60d:", round(c60[mask].mean(), 4),
      " 120d:", round(c120[mask].mean(), 4))

# Cross-sectional dispersion: std of 20d returns
disp20 = r20.std()
print("20d cross-sectional dispersion (std of ret20d):", round(disp20, 3), "%")

# Max drawdown over last 60d per asset
dd = {}
for a in ASSETS:
    s = px[a].dropna().tail(60)
    if len(s) < 20:
        dd[a] = np.nan
        continue
    roll_max = s.cummax()
    d = (s / roll_max - 1).min() * 100
    dd[a] = d
print("\n60d max drawdown %:", {k: round(v, 2) for k, v in dd.items()})

# Which assets above MA20
print("\nabove MA20:", [a for a in ASSETS if c_ma20[a] > 1.0])
print("below MA20:", [a for a in ASSETS if c_ma20[a] < 1.0])

# frozen feed check (zero returns recent)
zr = ret.tail(20).abs().sum()
print("\n20d zero-return days per asset:", zr.round(1).to_dict())
