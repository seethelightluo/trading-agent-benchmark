"""miner_3 cycle 2030-05-30 regime probe (data visible through 2030-05-29)."""
import pandas as pd
import numpy as np

VISIBLE = "2030-05-29"
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

def load_close(sym, cutoff, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date").sort_index()

px = pd.DataFrame({s: load_close(s, VISIBLE)["close"].astype(float) for s in TRADABLE})
ret = px.pct_change()

print("=== regime probe as of", VISIBLE, "===")
print("last close date:", px.index[-1].date(), "n_days:", len(px))
for w in [5, 20, 60, 120]:
    r = px.iloc[-1] / px.iloc[-1 - w] - 1
    print(f"\n--- {w}d return ---")
    for s in TRADABLE:
        print(f"  {s:10s} {r[s]*100:8.2f}%")

obs = {s: load_close(s, VISIBLE, INDEX_DIR)["close"].astype(float) for s in OBS}
print("\n--- observation signals ---")
for s in OBS:
    v = obs[s]
    print(f"  {s:8s} last={v.iloc[-1]:.2f} 5d={v.iloc[-1]/v.iloc[-6]-1:+.2%} 20d={v.iloc[-1]/v.iloc[-21]-1:+.2%} 60d={v.iloc[-1]/v.iloc[-61]-1:+.2%}")

# cross-sectional dispersion
print("\n--- dispersion: std of 20d returns across assets ---")
print("  20d cross-section std:", round(ret.iloc[-20:].sum().std(), 4))
print("  10d cross-section std:", round(ret.iloc[-10:].sum().std(), 4))

# vol levels
print("\n--- annualized 20d vol ---")
v20 = ret.tail(21).std() * np.sqrt(252)
for s in sorted(v20, key=v20.get, reverse=True):
    print(f"  {s:10s} {v20[s]*100:6.1f}%")

# 60d max drawdown per asset
print("\n--- 60d max drawdown ---")
for s in TRADABLE:
    c = px[s].tail(61)
    dd = (c / c.cummax() - 1).min()
    print(f"  {s:10s} {dd*100:7.2f}%")
