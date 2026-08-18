"""Screener analysis for cycle ending 2030-02-07 (visible horizon 2030-02-06)."""
import json, base64, zlib, io, os
import pandas as pd
import numpy as np

HORIZON = "2030-02-06"
ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
OBS = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

# ---------- 1. Load price data, cut at horizon ----------
px = {}
for a in ASSETS:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= HORIZON].set_index("date")["close"].rename(a)
    px[a] = df
px = pd.DataFrame(px)

obs = {}
for o in OBS:
    df = pd.read_csv(f"../persistent/index_data/{o}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= HORIZON].set_index("date")["close"].rename(o)
    obs[o] = df
obs = pd.DataFrame(obs)

print("=== PRICE PANEL ===")
print("rows:", len(px), "cols:", len(px.columns), "range:", px.index.min().date(), "->", px.index.max().date())

# ---------- 2. Returns ----------
ret = px.pct_change()
logret = np.log(px / px.shift(1))

# ---------- 3. Regime metrics ----------
last = px.iloc[-1]
ma20 = px.tail(20).mean()
ma60 = px.tail(60).mean()
ma120 = px.tail(120).mean()

print("\n=== LEVELS (as of 2030-02-06) ===")
for a in ASSETS:
    print(f"{a:10s} close={last[a]:10.2f}  ma20={ma20[a]:10.2f}  ma60={ma60[a]:10.2f}  ma120={ma120[a]:10.2f}")

# 20d / 60d / 120d returns
r20 = (px.iloc[-1] / px.iloc[-21] - 1) * 100
r60 = (px.iloc[-1] / px.iloc[-61] - 1) * 100
r120 = (px.iloc[-1] / px.iloc[-121] - 1) * 100
print("\n=== RETURNS (pct) ===")
for a in ASSETS:
    print(f"{a:10s} r20={r20[a]:+7.2f}  r60={r60[a]:+7.2f}  r120={r120[a]:+7.2f}")

# Volatility
vol20 = ret.tail(20).std() * np.sqrt(252) * 100
vol60 = ret.tail(60).std() * np.sqrt(252) * 100
print("\n=== ANNUALIZED VOL (pct) ===")
for a in ASSETS:
    print(f"{a:10s} vol20={vol20[a]:6.2f}  vol60={vol60[a]:6.2f}")

# Cross-sectional dispersion
cs_disp20 = ret.tail(20).std(axis=1).mean() * np.sqrt(252) * 100
cs_disp60 = ret.tail(60).std(axis=1).mean() * np.sqrt(252) * 100
print(f"\ncross-sectional dispersion (avg daily cross-sectional std, ann.): 20d={cs_disp20:.2f}%  60d={cs_disp60:.2f}%")

# Average pairwise correlation over 60d
c = ret.tail(60).corr()
mask = np.triu(np.ones(c.shape, dtype=bool), k=1)
avg_corr = c.values[mask].mean()
print(f"avg pairwise correlation 60d: {avg_corr:.3f}")

# ---------- 4. Macro observation ----------
print("\n=== MACRO OBSERVATION (last 5) ===")
for o in OBS:
    v = obs[o].dropna()
    print(f"{o:8s} last={v.iloc[-1]:8.2f}  20d chg={((v.iloc[-1]/v.iloc[-21]-1)*100) if len(v)>21 else float('nan'):+6.2f}%  60d chg={((v.iloc[-1]/v.iloc[-61]-1)*100) if len(v)>61 else float('nan'):+6.2f}%")

# VIX regime
vix = obs["VIX"].dropna()
vix_ma20 = vix.tail(20).mean()
print(f"\nVIX last={vix.iloc[-1]:.1f}  ma20={vix_ma20:.1f}  max60={vix.tail(60).max():.1f}  min60={vix.tail(60).min():.1f}")

# ---------- 5. Trend classification ----------
print("\n=== TREND / REGIME CLASSIFICATION ===")
spx = px["SPX"]
spx_ma20 = spx.tail(20).mean(); spx_ma60 = spx.tail(60).mean(); spx_ma120 = spx.tail(120).mean()
print(f"SPX close={spx.iloc[-1]:.0f} vs ma20={spx_ma20:.0f} ma60={spx_ma60:.0f} ma120={spx_ma120:.0f}")
print(f"SPX 20d={r20['SPX']:+.1f}% 60d={r60['SPX']:+.1f}% 120d={r120['SPX']:+.1f}%")

# drawdown from 120d high
dd = (px / px.tail(120).max() - 1) * 100
print("\n=== DRAWDOWN FROM 120D HIGH (pct) ===")
for a in ASSETS:
    print(f"{a:10s} dd120={dd[a].iloc[-1]:+6.2f}%")

# Best/worst performers over 60d
print("\n=== TOP/BOTTOM 60d PERFORMERS ===")
print("top:", r60.sort_values(ascending=False).head(5).round(2).to_dict())
print("bot:", r60.sort_values(ascending=True).head(5).round(2).to_dict())

# ---------- 6. Factor library summary ----------
print("\n=== FACTOR LIBRARY (active .json) ===")
factor_files = [f for f in os.listdir("factors") if f.endswith(".json") and not f.endswith(".bak") and f != "factor_ensemble.json"]
for f in sorted(factor_files):
    d = json.load(open(f"factors/{f}"))
    m = d.get("validation", {}).get("metrics", {})
    ic = m.get("ic", float("nan"))
    icir = m.get("icir", float("nan"))
    q = abs(ic) * abs(icir) if ic == ic and icir == icir else float("nan")
    to = m.get("turnover_10d_rank", float("nan"))
    tags = d.get("tags", [])
    ed = d.get("expected_direction", "")
    print(f"{d['factor_id']:24s} ic={ic:+.4f} icir={icir:+.3f} q={q:.5f} turn={to:.3f} dir={ed} tags={tags}")

# save a summary for later
summary = {
    "levels": {a: float(last[a]) for a in ASSETS},
    "r20": r20.to_dict(), "r60": r60.to_dict(), "r120": r120.to_dict(),
    "vol20": vol20.to_dict(), "vol60": vol60.to_dict(),
    "cs_disp20": float(cs_disp20), "cs_disp60": float(cs_disp60),
    "avg_corr60": float(avg_corr),
    "vix_last": float(vix.iloc[-1]), "vix_ma20": float(vix_ma20),
    "dd120": dd.iloc[-1].to_dict(),
}
with open("scripts/screener_summary_20300207.json", "w") as fh:
    json.dump(summary, fh, indent=1)
print("\nsaved scripts/screener_summary_20300207.json")
