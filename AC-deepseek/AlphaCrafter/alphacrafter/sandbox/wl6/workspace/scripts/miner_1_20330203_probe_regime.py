"""miner_1 2033-02-03: market regime + frozen-asset snapshot as of visible 2033-02-02."""
import pandas as pd
import numpy as np

VISIBLE = "2033-02-02"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

def load(sym, ddir="../persistent/stock_data", cutoff=VISIBLE):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)].set_index("date").sort_index()
    return df

px = pd.DataFrame({s: load(s)["close"].astype(float) for s in TRADABLE})
obs = {s: load(s, "../persistent/index_data")["close"].astype(float) for s in OBS}

ret = px.pct_change()
frozen = [s for s in TRADABLE if ret[s].dropna().iloc[-250:].abs().max() < 1e-12 or px[s].nunique() <= 1]
active = [s for s in TRADABLE if s not in frozen]
print("frozen:", frozen, "| active:", len(active))
print("last date:", px.index.max().date(), "| rows:", len(px))

rows = []
for s in TRADABLE:
    if s in frozen:
        rows.append([s, 0, 0, 0, 0, 0, 0])
        continue
    r = px[s]
    def rret(w, skip=0):
        return r.shift(skip) / r.shift(skip + w) - 1.0
    vol20 = ret[s].rolling(20).std() * np.sqrt(252)
    vol60 = ret[s].rolling(60).std() * np.sqrt(252)
    dist60high = r / r.rolling(60).max() - 1.0
    rows.append([s, rret(5), rret(20), rret(60), rret(120), vol20, vol60, dist60high])

tab = pd.DataFrame(rows, columns=["sym", "r5", "r20", "r60", "r120", "vol20", "vol60", "dist60high"])
print(tab.round(4).to_string(index=False))

print()
for o in OBS:
    v = obs[o]
    print(f"{o}: last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.3f} 60d={v.iloc[-1]/v.iloc[-61]-1:+.3f} mean60={v.iloc[-60:].mean():.2f} min60={v.iloc[-60:].min():.2f} max60={v.iloc[-60:].max():.2f}")

cross20 = ret[active].std() * np.sqrt(252)
print("\ncross-sectional annualized vol (active):", cross20.round(4).mean())