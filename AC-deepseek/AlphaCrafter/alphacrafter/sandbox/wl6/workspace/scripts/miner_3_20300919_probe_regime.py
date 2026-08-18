"""miner_3 2030-09-19 regime probe. Data visible through 2030-09-18 (last completed day)."""
import pandas as pd
import numpy as np

VISIBLE = "2030-09-18"
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
print("last date:", px.index.max().date(), "| n rows:", len(px), flush=True)

# only use rows through visible
ret = px.pct_change()
frozen = [s for s in TRADABLE if px[s].nunique() <= 1 or ret[s].dropna().abs().max() < 1e-12]
active = [s for s in TRADABLE if s not in frozen]
print("frozen:", frozen, "| active:", len(active), flush=True)

def stats(sym, w=20):
    r = ret[sym].dropna()
    if len(r) < w:
        return None
    last = px[sym].iloc[-1]
    rw = r.iloc[-w:]
    return dict(last=round(last, 2), r20=round(px[sym].iloc[-1] / px[sym].iloc[-1 - w] - 1, 4),
                r60=round(px[sym].iloc[-1] / px[sym].iloc[-1 - 60] - 1, 4) if len(px) > 60 else np.nan,
                vol20=round(rw.std() * np.sqrt(252), 4), mom5=round(px[sym].iloc[-1] / px[sym].iloc[-6] - 1, 4))

print("\n=== per-asset stats (through %s) ===" % VISIBLE)
for s in TRADABLE:
    st = stats(s)
    if st:
        print(f"{s:<10} last={st['last']:>10.2f} r5={st['mom5']:>8.2%} r20={st['r20']:>8.2%} r60={st['r60']:>8.2%} vol20={st['vol20']:>6.2%}", flush=True)

# observation-only signals
obs = {s: load_close(s, VISIBLE, INDEX_DIR)["close"].astype(float) for s in OBS}
print("\n=== macro obs ===")
for s in OBS:
    o = obs[s]
    o20 = o.iloc[-1] / o.iloc[-21] - 1 if len(o) > 21 else np.nan
    o60 = o.iloc[-1] / o.iloc[-61] - 1 if len(o) > 61 else np.nan
    print(f"{s:<8} last={o.iloc[-1]:>9.2f} r20={o20:>8.2%} r60={o60:>8.2%}", flush=True)

# dispersion
r20 = ret.iloc[-20:].apply(lambda x: (1 + x).prod() - 1)
print("\n20d cross-sectional dispersion: max-min = %.2f%% | std = %.2f%%" %
      ((r20.max() - r20.min()) * 100, r20.std() * 100), flush=True)
print("\nTop 5 20d:", r20.nlargest(5).round(4).to_dict(), flush=True)
print("Bot 5 20d:", r20.nsmallest(5).round(4).to_dict(), flush=True)

# VIX context
vix = obs['VIX']
print("\nVIX last=%.1f 5d ago=%.1f 60d max=%.1f min=%.1f mean=%.1f" %
      (vix.iloc[-1], vix.iloc[-6], vix.iloc[-60:].max(), vix.iloc[-60:].min(), vix.iloc[-60:].mean()), flush=True)

# US10Y / CN10Y context
print("US10Y last=%.3f r60=%.2f%% | CN10Y last=%.3f r60=%.2f%%" %
      (px['US10Y'].iloc[-1], (px['US10Y'].iloc[-1]/px['US10Y'].iloc[-61]-1)*100,
       px['CN10Y'].iloc[-1], (px['CN10Y'].iloc[-1]/px['CN10Y'].iloc[-61]-1)*100), flush=True)

# trend classification
spx_trend = px['SPX'].iloc[-1] / px['SPX'].iloc[-61] - 1 if len(px) > 60 else 0
if spx_trend > 0.05:
    regime = "bull"
elif spx_trend < -0.05:
    regime = "bear"
else:
    regime = "sideways"
print(f"\nregime(SPX 60d)={regime} trend={spx_trend:.2%} VIX={vix.iloc[-1]:.1f}", flush=True)
