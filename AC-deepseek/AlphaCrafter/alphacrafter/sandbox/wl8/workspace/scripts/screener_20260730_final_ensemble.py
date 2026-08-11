"""Factor screener - 2026-07-30 final ensemble.

Base: quality_IC_tilt  (q = |IC| * |ICIR| * min(1, n_ic_dates/500), direction = sign(IC))
Overlay: regime-stability multiplier from recent-window (6m) revalidation IC
alignment with the persisted direction; persistent recent sign-flips demote the
factor but do NOT flip direction (full-sample n >> recent n).

Market regime (data through 2026-07-29, previous completed trading day):
  - Trend: mixed/corrective, high dispersion, no single bull/bear.
  - Risk: Medium (VIX 15.8, low-moderate) with high-vol pockets (SOX 61%,
    ETH 52%, STAR50 49%, WTI 45% ann. realized).
  - Correlation: LOW cross-asset (median pairwise 60d rho ~0.10, median |rho| ~0.21)
    -> cross-sectional dispersion ~11% (21d CS std) favors cross-sectional factors.
  - Rates: US10Y yield rising (~+3% 1M); DXY weak ~100.4 (-1% 1M).
"""
import json, base64, zlib, io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

END = "2026-07-29"
ASSETS = ["000300.SH", "000688.SH", "SPX", "NDX", "SOX", "HSI", "N225", "SX5E",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTORS = ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]

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

close = load_close(ASSETS)
panels = {f: lib_panel(f) for f in FACTORS}

# ---- persisted metrics + recent-window (6m) revalidation ----
meta = {}
recent = {}
for f in FACTORS:
    d = json.load(open(f"factors/{f}.json"))
    m = d["validation"]["metrics"]
    meta[f] = dict(ic=m["ic"], icir=m["icir"], n=m["n_ic_dates"],
                   cov=m["coverage_asset_days"], to=m["turnover_10d_rank"],
                   expected_direction=d.get("expected_direction", 1))
    r = rank_ic(panels[f], close, start="2026-01-30")   # ~6m
    r6 = rank_ic(panels[f], close, start="2025-07-30")  # ~1y
    recent[f] = dict(six_m=r, one_y=r6)
    print(f"{f:24s} persisted IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} n={m['n_ic_dates']:5d} | "
          f"6m IC={r['ic']:+.4f} (n={r['n']:3d}) | 1y IC={r6['ic']:+.4f} (n={r6['n']:3d})")

# ---- stability multiplier: sign agreement with persisted direction ----
STAB = {}   # factor -> (multiplier, reason)
for f in FACTORS:
    r6 = recent[f]["six_m"]; r1 = recent[f]["one_y"]
    pers_dir = np.sign(meta[f]["ic"])
    agree = 0
    if r6 is not None and np.sign(r6["ic"]) == pers_dir:
        agree += 1
    if r1 is not None and np.sign(r1["ic"]) == pers_dir:
        agree += 1
    # 0/2 or 1/2 agreement with weak recent ICIR -> demote; 2/2 -> keep
    if agree == 2:
        STAB[f] = (1.0, "recent-window IC aligned with persisted direction")
    elif agree == 1:
        STAB[f] = (0.8, "mixed recent-window IC alignment (one window flipped)")
    else:
        STAB[f] = (0.6, "recent-window IC flipped vs persisted direction")

print("\n=== quality-IC-tilt + stability overlay ===")
rows = []
for f in FACTORS:
    ic = meta[f]["ic"]; icir = meta[f]["icir"]; n = meta[f]["n"]
    shrink = min(1.0, n / 500.0)
    q_base = abs(ic) * abs(icir) * shrink
    mult, why = STAB[f]
    q = q_base * mult
    rows.append(dict(factor_id=f, ic=ic, icir=icir, n=n, shrink=shrink,
                     q_base=q_base, mult=mult, q=q,
                     direction=1 if ic >= 0 else -1, stab_reason=why))
tot = sum(r["q"] for r in rows)
out = []
for r in rows:
    w = r["q"] / tot
    r["weight"] = w
    out.append({"factor_id": r["factor_id"], "weight": round(w, 4), "direction": r["direction"]})
    print("  %-22s IC=%+.4f ICIR=%+.4f n=%4d shrink=%.3f q=%.6f stab=%.2f -> w=%.4f dir=%+d  [%s]"
          % (r["factor_id"], r["ic"], r["icir"], r["n"], r["shrink"], r["q"], r["mult"], w,
             r["direction"], r["stab_reason"]))
print("sum(w) =", round(sum(r["weight"] for r in rows), 6))

# ---- persist ----
ensemble = {
    "schema_version": 1,
    "selected_factors": out,
    "method": "quality_ic_tilt_with_regime_stability_overlay",
    "notes": {
        "asof": "2026-07-30",
        "quality_formula": "q = |IC| * |ICIR| * min(1, n_ic_dates/500); direction = sign(IC); "
                           "stability multiplier 0.6/0.8/1.0 from 6m+1y revalidation IC sign agreement",
        "library": {f: meta[f] for f in FACTORS},
        "recent_revalidation": {
            f: {"six_m_ic": recent[f]["six_m"]["ic"], "six_m_n": recent[f]["six_m"]["n"],
                "one_y_ic": recent[f]["one_y"]["ic"], "one_y_n": recent[f]["one_y"]["n"]}
            for f in FACTORS},
        "regime": ("mixed/corrective, HIGH dispersion, LOW cross-asset correlation "
                   "(60d median pairwise rho ~0.10, median |rho| ~0.21): SPX +1.7% 1M resilient; "
                   "CSI300 -17% 1M sharp bear leg; HSI +16% 1M V-recovery; SOX -17% 1M semis correction; "
                   "ETH -20% 1M / XAU -10% 1M weak; WTI +9% 1M bounce after -29% 3M; US10Y yield rising; "
                   "DXY weak 100.4; VIX 15.8 low-moderate (falling 1M)"),
        "risk_notes": {
            "mom_10d_skip5": "TO_10d_rank=4.09 highest turnover -> expect larger rebalance migration (3bps on migrated notional); sharp 1M reversals (HSI +16%, WTI +9%) raise whipsaw risk, but recent IC strongest of the three",
            "vix_beta_cond_60x20": "recent-window IC flipped positive (6m +0.024, 1y +0.011) vs persisted -0.038; VIX low/falling regime may invalidate short-vix-beta tilt; direction kept at persisted sign (n=1600 full-sample), weight demoted; TRADER: monitor PnL attribution",
            "yield_beta_cond_60x20": "sparse evidence (n=49 IC dates, 14.6% asset-day coverage, only 2% dates with >=8 assets); recent IC negative on tiny samples (1y n=5); heavy evidence shrink applied; keep small diversifying weight",
            "crowding": "max |pairwise panel corr| = 0.344 (mom vs yield), all < 0.5 gate; no redundancy; vix_beta decorrelated (|rho|<=0.17)"
        },
        "evidence_note": ("mom_10d_skip5 is the regime-favored primary (improving IC across 2025+/1y/6m, "
                          "hit ratio ~0.56-0.59). vix_beta and yield_beta kept as diversifiers with demoted "
                          "weights due to recent sign instability / sparse evidence respectively.")
    }
}
with open("factor_ensemble.json", "w") as fh:
    json.dump(ensemble, fh, indent=2)
print("\n[wrote] factor_ensemble.json")
