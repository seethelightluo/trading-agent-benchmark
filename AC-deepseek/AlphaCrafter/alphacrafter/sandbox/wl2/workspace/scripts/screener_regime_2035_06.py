"""Screener cycle 2035-06-21: compute regime metrics + 5 admitted factor exposures
as of visible_through=2035-06-20. Uses only data <= 2035-06-20."""
import json
import numpy as np
import pandas as pd

VIS = "2035-06-20"
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def load_close(sym, root="../persistent/stock_data/"):
    df = pd.read_csv(f"{root}{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VIS].set_index("date")["close"].astype(float)
    return df

closes = {a: load_close(a) for a in ASSETS}

# flat-feed detection (trailing 15d std)
print("=== flat-feed detection (trailing 15d return std == 0) ===")
flat = []
for a in ASSETS:
    c = closes[a]
    s = c.pct_change().dropna().tail(15)
    isflat = bool(len(s) >= 15 and s.std() < 1e-12)
    if isflat:
        flat.append(a)
    print(f"{a:10s} n={len(c):5d} last={c.iloc[-1]:12.4f} r15std={s.std():.6f} flat={isflat}")
print("FLAT:", flat)

# ============ factor exposures ============
spx_ret = closes["SPX"].pct_change()
factors = {fid: [] for fid in ["max_consec_gain_20", "mom_180d_skip5", "range_pos_252", "spx_corr60", "downbeta_spx_60"]}

def longest_run(x):
    m = 0.0; cur = 0
    for v in x:
        if v == 1:
            cur += 1; m = max(m, cur)
        else:
            cur = 0
    return m

per = {}
for a in ASSETS:
    c = closes[a]
    ret = c.pct_change()
    pos = (ret > 0).astype(int)
    m = {}
    m["max_consec_gain_20"] = pos.rolling(21, min_periods=10).apply(longest_run, raw=True).iloc[-1]
    m["mom_180d_skip5"] = (c.shift(5) / c.shift(185) - 1.0).iloc[-1]
    rmin = c.rolling(252, min_periods=30).min(); rmax = c.rolling(252, min_periods=30).max()
    m["range_pos_252"] = ((c - rmin) / (rmax - rmin).replace(0, np.nan)).iloc[-1]
    if len(ret) >= 60:
        m["spx_corr60"] = ret.rolling(60, min_periods=15).corr(spx_ret).iloc[-1]
        m2 = pd.concat([ret, spx_ret], axis=1, join="inner").dropna(); m2.columns = ["a", "s"]
        sub = m2[m2["s"] < 0].tail(60)
        if len(sub) >= 15 and sub["s"].var() > 1e-12:
            m["downbeta_spx_60"] = float(sub["a"].cov(sub["s"]) / sub["s"].var())
        else:
            m["downbeta_spx_60"] = np.nan
    else:
        m["spx_corr60"] = np.nan; m["downbeta_spx_60"] = np.nan
    per[a] = m

print("\n=== factor exposures as of", VIS, "===")
hdr = f"{'asset':10s} {'maxcg':>7s} {'mom180':>8s} {'rng252':>7s} {'corr60':>7s} {'dnbeta':>8s}"
print(hdr)
for a in ASSETS:
    m = per[a]
    def fmt(v):
        return "   NaN " if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:7.3f}"
    print(f"{a:10s} {fmt(m['max_consec_gain_20'])} {fmt(m['mom_180d_skip5'])} {fmt(m['range_pos_252'])} {fmt(m['spx_corr60'])} {fmt(m['downbeta_spx_60'])}")

# ============ returns snapshot ============
print("\n=== returns snapshot (r5/r20/r60/r252) ===")
print(f"{'asset':10s} {'r5':>8s} {'r20':>8s} {'r60':>8s} {'r252':>9s} {'vol20':>7s}")
for a in ASSETS:
    c = closes[a]
    def r(n):
        if len(c) > n:
            return c.iloc[-1] / c.iloc[-1 - n] - 1.0
        return np.nan
    s = c.pct_change().dropna().tail(20)
    v = float(s.std()) if len(s) >= 5 else np.nan
    print(f"{a:10s} {r(5)*100:8.2f} {r(20)*100:8.2f} {r(60)*100:8.2f} {r(252)*100:9.2f} {v*100:7.1f}")

# ============ macro ============
print("\n=== macro (observation only) ===")
for sym in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    df = pd.read_csv(f"../persistent/index_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VIS].set_index("date")["close"].astype(float)
    def r(n):
        return (df.iloc[-1] / df.iloc[-1 - n] - 1.0) * 100 if len(df) > n else np.nan
    print(f"{sym:8s} last={df.iloc[-1]:10.3f} mean60={df.tail(60).mean():9.3f} r5={r(5):6.2f} r20={r(20):6.2f} r60={r(60):6.2f}")
