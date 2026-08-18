"""miner_2 2027-06-17: debug data quality issues found in 06-03 exploration.

Issue 1: cross-asset beta factors (xau/wti/btc/us10y beta_60) have cov=0.070, nA=0
         but IC n=174 all within recent 12m. Why are valid values clustered recently?
Issue 2: pv_corr_20 has empty recent window (R12m n=39, R6m n=0). Volume data issue?
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def load_asset(symbol, days=3200):
    df = None
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        try:
            df = get_stock_daily_data(symbol=symbol, days=days)
        except Exception:
            df = None
    if df is None or len(df) < 400:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


print("loading ...")
data = {a: load_asset(a) for a in WATCH}
data = {a: d for a, d in data.items() if d is not None}
closes = {a: d["close"].astype(float) for a, d in data.items()}
vols = {a: (d["volume"].astype(float) if "volume" in d.columns and d["volume"].notna().any() else None)
        for a, d in data.items()}
fclose = pd.DataFrame(closes).sort_index()
fvol = pd.DataFrame({a: v for a, v in vols.items() if v is not None}).sort_index()
rets = fclose.pct_change()
print(f"fclose {fclose.shape}, fvol {fvol.shape}, last={fclose.index.max().date()}")

# ---- Issue 1: xau_beta_60 valid counts by year ----
rxau = rets["XAU"]
xau_beta = {}
for a, c in closes.items():
    r = c.pct_change()
    xau_beta[a] = r.rolling(60).cov(rxau) / rxau.rolling(60).var()
xdf = pd.DataFrame(xau_beta)
print("\n=== xau_beta_60 valid asset-days per year ===")
for yr in range(2020, 2028):
    sub = xdf.loc[xdf.index.year == yr]
    print(f"{yr}: dates={len(sub):4d} valid_cells={int(sub.notna().sum().sum()):5d} "
          f"avg_assets_per_date={sub.notna().sum(axis=1).mean():.1f}")
print("assets with >200 valid obs:", int((xdf.notna().sum() > 200).sum()))
print("per-asset valid counts:", dict(xdf.notna().sum()))

# ---- Issue 2: volume data recency ----
print("\n=== volume panel recent availability ===")
print("volume last non-null date per asset:")
for a in fvol.columns:
    s = fvol[a]
    last = s[s.notna() & (s > 0)].index.max()
    n_last_250 = int((s.loc[s.index >= "2026-06-01"] > 0).sum())
    print(f"  {a:10s} last_valid={last.date() if last is not None else None} "
          f"valid_days_since_2026-06={n_last_250}")

# ---- pv_corr_20 valid counts by year ----
print("\n=== pv_corr_20 valid asset-days per year ===")
pv = {}
for a in fvol.columns:
    rc = fclose[a].pct_change()
    v = fvol[a]
    pv[a] = rc.rolling(20).corr(v)
pdf = pd.DataFrame(pv)
for yr in range(2020, 2028):
    sub = pdf.loc[pdf.index.year == yr]
    print(f"{yr}: dates={len(sub):4d} valid_cells={int(sub.notna().sum().sum()):5d} "
          f"avg_assets_per_date={sub.notna().sum(axis=1).mean():.1f}")

# ---- check returns: are some assets flat recently? ----
print("\n=== returns nonzero share in last 250 days ===")
r250 = rets.loc[rets.index >= "2026-06-01"]
for a in fclose.columns:
    s = r250[a].dropna()
    nz = (s != 0).mean()
    print(f"  {a:10s} nonzero_ret_share={nz:.3f} n={len(s)}")
