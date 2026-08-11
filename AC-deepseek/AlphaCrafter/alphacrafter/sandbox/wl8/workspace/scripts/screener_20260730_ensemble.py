"""Factor screener - 2026-07-30 cycle.

Loads the persisted live library (factors/*.json, non-bak), re-validates
recent-window rank IC on the 15-asset universe, checks internal factor
correlation, and builds the quality-IC-tilt ensemble with evidence shrink.
"""
import json, base64, zlib, io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

END = "2026-07-29"
ASSETS = ["000300.SH", "000688.SH", "SPX", "NDX", "SOX", "HSI", "N225", "SX5E",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def load_close(assets, end=END):
    closes = {}
    for a in assets:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= end]
        closes[a] = df.set_index("date")["close"].astype(float)
    return pd.DataFrame(closes)

def lib_panel(fname):
    d = json.load(open(f"factors/{fname}.json"))
    art = d["validation"]["signal_artifact"]
    csv = zlib.decompress(base64.b64decode(art["data"])).decode()
    p = pd.read_csv(io.StringIO(csv), index_col=0)
    p.index = pd.to_datetime(p.index)
    return p

def rank_ic(fdf, close, h=10, min_assets=8, start=None, end=None):
    fwd = close.shift(-h) / close - 1.0
    common = fdf.index.intersection(fwd.index)
    if start:
        common = common[common >= pd.Timestamp(start)]
    if end:
        common = common[common <= pd.Timestamp(end)]
    ics = []
    for d in common:
        f = fdf.loc[d].dropna(); r = fwd.loc[d].dropna()
        both = f.index.intersection(r.index)
        if len(both) >= min_assets:
            ic = spearmanr(f[both], r[both])[0]
            if np.isfinite(ic):
                ics.append(ic)
    if len(ics) < 5:
        return None
    a = np.array(ics)
    return dict(n=len(a), ic=float(a.mean()),
                icir=float(a.mean()/a.std(ddof=1)) if len(a) > 2 else 0.0,
                hit=float((a > 0).mean()))

def panel_corr(fa, fb):
    sa = fa.stack(); sb = fb.stack()
    both = sa.index.intersection(sb.index)
    sa = sa.loc[both]; sb = sb.loc[both]
    m = sa.notna() & sb.notna()
    if m.sum() < 100:
        return float("nan")
    return float(np.corrcoef(sa[m], sb[m])[0, 1])

close = load_close(ASSETS)
print("close range:", close.index.min().date(), "->", close.index.max().date())

factors = ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]
panels = {f: lib_panel(f) for f in factors}
meta = {}
for f in factors:
    d = json.load(open(f"factors/{f}.json"))
    m = d["validation"]["metrics"]
    meta[f] = dict(ic=m["ic"], icir=m["icir"], n=m["n_ic_dates"],
                   cov=m["coverage_asset_days"], to=m["turnover_10d_rank"],
                   expected_direction=d.get("expected_direction", 1))
    print(f"\n=== {f} ===")
    print("  persisted: IC=%.4f ICIR=%.4f n=%d cov=%.3f turn=%.2f dir=%+d" %
          (m["ic"], m["icir"], m["n_ic_dates"], m["coverage_asset_days"],
           m["turnover_10d_rank"], d.get("expected_direction", 1)))
    for label, (s, e) in {"full": (None, None),
                          "2025+": ("2025-01-01", None),
                          "1y": ("2025-07-30", None),
                          "6m": ("2026-01-30", None)}.items():
        r = rank_ic(panels[f], close, start=s, end=e)
        if r:
            print("  %-5s IC=%+.4f ICIR=%+.3f n=%d hit=%.2f" % (label, r["ic"], r["icir"], r["n"], r["hit"]))

print("\n=== internal panel correlations ===")
for i in range(len(factors)):
    for j in range(i+1, len(factors)):
        print(f"  {factors[i]} vs {factors[j]}: rho={panel_corr(panels[factors[i]], panels[factors[j]]):+.3f}")

# ---------------- ensemble construction ----------------
print("\n=== quality-IC-tilt weights ===")
rows = []
for f in factors:
    ic = meta[f]["ic"]; icir = meta[f]["icir"]; n = meta[f]["n"]
    shrink = min(1.0, n / 500.0)
    q = abs(ic) * abs(icir) * shrink
    rows.append(dict(factor_id=f, ic=ic, icir=icir, n=n, shrink=shrink, q=q,
                     direction=1 if ic >= 0 else -1))
tot = sum(r["q"] for r in rows)
out = []
for r in rows:
    w = r["q"] / tot if tot > 0 else 0.0
    r["weight"] = w
    out.append({"factor_id": r["factor_id"], "weight": round(w, 4), "direction": r["direction"]})
    print("  %-22s IC=%+.4f ICIR=%+.4f n=%4d shrink=%.3f q=%.6f -> w=%.4f dir=%+d" %
          (r["factor_id"], r["ic"], r["icir"], r["n"], r["shrink"], r["q"], w, r["direction"]))
print("sum(w) =", round(sum(r["weight"] for r in rows), 4))

# persist ensemble
ensemble = {
    "schema_version": 1,
    "selected_factors": out,
    "method": "quality_ic_tilt",
    "notes": {
        "asof": "2026-07-30",
        "quality_formula": "q = |IC| * |ICIR| * min(1, n_ic_dates/500); weights = q/sum(q); direction = sign(IC)",
        "library": {f: meta[f] for f in factors},
        "regime": "mixed/corrective, high dispersion, LOW cross-asset correlation (median |rho| 60d ~0.14), falling VIX (14.1, -20% 1m): SPX +2.8% 1m resilient bull; CSI300 -17% 1m sharp bear leg; HSI +15% 1m V-recovery; XAU -19% 3m correction; ETH -31% 3m weak; US10Y yield +6.5% 1m rising; DXY weak 99.7",
        "evidence_note": "yield_beta_cond_60x20 IC high but n=49 IC dates (sparse 14.6% coverage) -> heavy evidence shrink; mom_10d_skip5 turnover_10d_rank=4.09 high -> expect larger rebalance migration"
    }
}
with open("factor_ensemble.json", "w") as fh:
    json.dump(ensemble, fh, indent=2)
print("\n[wrote] factor_ensemble.json")
