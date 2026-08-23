"""Full validation for range_20_neg (gate PASS from screen). Compute decay,
turnover, coverage, library correlation, and recency split. Decide persistence.
"""
import sys, os, json, base64, zlib
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE, ic_analysis, library_corr
import pandas as pd, numpy as np, math, glob

VIS = "2034-09-13"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"]); df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill().dropna(how="all").dropna(axis=1, how="all")
ret = px.pct_change()

hi = px.rolling(20).max(); lo = px.rolling(20).min()
rng = hi/lo - 1.0
factor = -rng  # range_20_neg

res = ic_analysis(factor, px, horizon=10, label="range_20_neg")
for k,v in res.items():
    if k not in ("decay_ic_by_horizon",): print(f"  {k}: {v}")
print("  decay_ic_by_horizon:", res["decay_ic_by_horizon"])

# library correlation: load persisted library signal artifacts if present
lib = {}
libdir = "factors"
for path in glob.glob(f"{libdir}/*.json"):
    if path.endswith(".bak") or "evicted" in path or "quarantine" in path: continue
    try:
        d = json.load(open(path))
    except Exception: continue
    sa = d.get("validation",{}).get("signal_artifact",{})
    if not sa or sa.get("format") != "base64:zlib:csv": continue
    raw = base64.b64decode(sa["data"])
    csv = zlib.decompress(raw).decode()
    sig = pd.read_csv(pd.io.common.StringIO(csv), index_col=0, parse_dates=True)
    lib[d["factor_id"]] = sig
print("library signals loaded:", len(lib))
if lib:
    mlc = library_corr(factor, lib)
    print("  max_abs_library_correlation:", round(mlc,3))

# recency split
ic10 = rank_ic_series(factor, align_fwd_returns(px,10))
for start,lab in [("2020-01-01","full"),("2025-01-01","2025+"),("2032-09-01","recent2y"),("2033-09-01","recent1y")]:
    s = ic10[ic10.index>=start]
    if len(s):
        m=float(s.mean()); std=float(s.std(ddof=1)) if len(s)>1 and s.std(ddof=1)>0 else np.nan
        print(f"  IC[{lab}] n={len(s)} IC={m:+.4f} ICIR={m/std if std and math.isfinite(std) else np.nan:+.3f}")