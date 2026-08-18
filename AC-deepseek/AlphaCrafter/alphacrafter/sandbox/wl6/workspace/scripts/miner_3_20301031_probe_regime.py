"""miner_3 2030-10-31: probe current regime. Data visible through 2030-10-30."""
import numpy as np
import pandas as pd

VISIBLE = "2030-10-30"
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']


def load_close(sym, cutoff, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date").sort_index()


px = {s: load_close(s, VISIBLE)["close"].astype(float) for s in TRADABLE}
px = pd.DataFrame(px)
ret = px.pct_change()
obs = {s: load_close(s, VISIBLE, INDEX_DIR)["close"].astype(float) for s in OBS}

print("=== tradable stats (through %s) ===" % VISIBLE, flush=True)
for s in TRADABLE:
    last = px[s].iloc[-1]
    r5 = px[s].iloc[-1] / px[s].iloc[-6] - 1
    r20 = px[s].iloc[-1] / px[s].iloc[-21] - 1
    r60 = px[s].iloc[-1] / px[s].iloc[-61] - 1
    vol20 = ret[s].iloc[-20:].std() * np.sqrt(252)
    print(f"{s:<10} last={last:>10.3f} r5={r5:>8.2%} r20={r20:>8.2%} r60={r60:>8.2%} vol20={vol20:>6.2%}", flush=True)

print("\n=== macro obs ===", flush=True)
for s in OBS:
    o = obs[s]
    o20 = o.iloc[-1] / o.iloc[-21] - 1 if len(o) > 21 else np.nan
    o60 = o.iloc[-1] / o.iloc[-61] - 1 if len(o) > 61 else np.nan
    print(f"{s:<8} last={o.iloc[-1]:>9.2f} r20={o20:>8.2%} r60={o60:>8.2%}", flush=True)

r20 = ret.iloc[-20:].apply(lambda x: (1 + x).prod() - 1)
print("\n20d cross-sectional dispersion: max-min = %.2f%% | std = %.2f%%" %
      ((r20.max() - r20.min()) * 100, r20.std() * 100), flush=True)
print("Top 5 20d:", r20.nlargest(5).round(4).to_dict(), flush=True)
print("Bot 5 20d:", r20.nsmallest(5).round(4).to_dict(), flush=True)

vix = obs['VIX']
print("\nVIX last=%.1f 5d ago=%.1f 60d max=%.1f min=%.1f mean=%.1f" %
      (vix.iloc[-1], vix.iloc[-6], vix.iloc[-60:].max(), vix.iloc[-60:].min(), vix.iloc[-60:].mean()), flush=True)

US10Y_last = px['US10Y'].iloc[-1]
CN10Y_last = px['CN10Y'].iloc[-1]
print("US10Y last=%.3f r60=%.2f%% | CN10Y last=%.3f r60=%.2f%%" %
      (US10Y_last, (US10Y_last / px['US10Y'].iloc[-61] - 1) * 100,
       CN10Y_last, (CN10Y_last / px['CN10Y'].iloc[-61] - 1) * 100), flush=True)

spx_trend = px['SPX'].iloc[-1] / px['SPX'].iloc[-61] - 1
if spx_trend > 0.05:
    regime = "bull"
elif spx_trend < -0.05:
    regime = "bear"
else:
    regime = "sideways"
print(f"\nregime(SPX 60d)={regime} trend={spx_trend:.2%} VIX={vix.iloc[-1]:.1f}", flush=True)

# frozen detection
frozen = [s for s in TRADABLE if ret[s].dropna().iloc[-250:].abs().max() < 1e-12 or px[s].nunique() <= 1]
print("frozen assets (flat/flat-history):", frozen, flush=True)