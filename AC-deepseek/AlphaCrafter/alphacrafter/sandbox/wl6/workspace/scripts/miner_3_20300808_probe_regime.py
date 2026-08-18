"""miner_3 cycle 2030-08-08 regime probe (data visible through 2030-08-07)."""
import pandas as pd
import numpy as np

VISIBLE = "2030-08-07"
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
for w in [5, 10, 20, 60, 120]:
    r = px.iloc[-1] / px.iloc[-1 - w] - 1
    print(f"\n--- {w}d return ---")
    for s in TRADABLE:
        print(f"  {s:10s} {r[s]*100:8.2f}%")

obs = {s: load_close(s, VISIBLE, INDEX_DIR)["close"].astype(float) for s in OBS}
print("\n--- observation signals ---")
for s in OBS:
    v = obs[s]
    print(f"  {s:8s} last={v.iloc[-1]:.2f} 5d={v.iloc[-1]/v.iloc[-6]-1:+.2%} 20d={v.iloc[-1]/v.iloc[-21]-1:+.2%} 60d={v.iloc[-1]/v.iloc[-61]-1:+.2%}")

print("\n--- dispersion: std of cumulative returns across assets ---")
for w in [5, 10, 20]:
    print(f"  {w}d cross-section std:", round(ret.iloc[-w:].sum().std(), 4))

print("\n--- annualized 20d vol ---")
vol20 = ret.rolling(20).std() * np.sqrt(252)
print(vol20.iloc[-1].sort_values().round(3).to_string())

# drawdown state
print("\n--- distance from 60d high ---")
d60 = (px.iloc[-1] / px.rolling(60, min_periods=30).max().iloc[-1] - 1)
print(d60.sort_values().round(4).to_string())
