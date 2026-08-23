"""miner_1 2034-07-25 probe: market regime + frozen asset snapshot as of visible 2034-07-24."""
import pandas as pd
import numpy as np

VISIBLE = "2034-07-24"
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
frozen = [s for s in TRADABLE if
          (len(ret[s].dropna()) and ret[s].dropna().iloc[-250:].abs().max() < 1e-12)
          or px[s].nunique() <= 1]
active = [s for s in TRADABLE if s not in frozen]
print("frozen:", frozen, "| active:", len(active))
print("last date:", px.index.max().date(), "| rows:", len(px))

rows = []
for s in TRADABLE:
    if s in frozen:
        rows.append([s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        continue
    r = px[s]
    def rret(w, skip=0):
        if len(r) <= w + skip:
            return np.nan
        return r.iloc[-1] / r.iloc[-1 - w - skip] - 1.0
    v20 = ret[s].iloc[-20:].std() * np.sqrt(252)
    v60 = ret[s].iloc[-60:].std() * np.sqrt(252)
    rows.append([s, rret(5), rret(20), rret(60), rret(120), v20, v60,
                 r.iloc[-1] / r.rolling(60).max().iloc[-1] - 1.0,
                 r.iloc[-1] / r.rolling(20).mean().iloc[-1] - 1.0])

tab = pd.DataFrame(rows, columns=["sym", "r5", "r20", "r60", "r120", "vol20", "vol60", "dist60high", "dist20ma"])
print(tab.round(4).to_string(index=False))

print()
for o in OBS:
    v = obs[o]
    print(f"{o}: last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.3f} 60d={v.iloc[-1]/v.iloc[-61]-1:+.3f} "
          f"mean60={v.iloc[-60:].mean():.2f} min60={v.iloc[-60:].min():.2f} max60={v.iloc[-60:].max():.2f}")

xsd20 = ret[active].iloc[-20:].std(axis=1).mean()
xsd60 = ret[active].iloc[-60:].std(axis=1).mean()
print(f"\ncross-sectional mean daily dispersion 20d={xsd20:.4f} 60d={xsd60:.4f}")

spx = px['SPX']
for w in [20, 60, 120, 200]:
    m = spx.rolling(w).mean()
    if len(m) > w:
        slope = (m.iloc[-1] / m.iloc[-1 - w] - 1) * 100
        print(f"SPX ma{w}={m.iloc[-1]:.0f} last={spx.iloc[-1]:.0f} above={spx.iloc[-1] > m.iloc[-1]} slope={slope:+.2f}%")