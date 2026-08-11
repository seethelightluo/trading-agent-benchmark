"""miner_1 persistence cycle 2026-07-30.
Re-validates candidates that passed the IC/ICIR admission gates in the
previous exploration cycle, checks redundancy vs the existing library AND
mutual correlation among candidates, then persists passing factors to
factors/ with complete JSON definitions + recoverable signal artifacts.

Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at h=10, max_abs_library_corr < 0.5.
"""
import sys, json, hashlib, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
macro = {
    "VIX": load_index("VIX"),
    "DXY": load_index("DXY"),
    "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"),
    "EURUSD": load_index("EURUSD"),
}
print(f"Panel dates {close.index[0].date()}..{close.index[-1].date()}, "
      f"{len(close)} rows, {close.shape[1]} assets")
lib = load_library_panels()
print(f"Library panels loaded: {list(lib.keys())}")

CS_MEAN_20 = close.pct_change(20).mean(axis=1)
CS_MEAN_60 = close.pct_change(60).mean(axis=1)
EW_RET = close.pct_change().mean(axis=1)
BTC_RET = close["BTC"].pct_change()

# ---------------- candidate definitions (per-asset dense series) ----------------
def f_rel_mom_20(c, v, o, h, l, m):
    r = c / c.shift(20) - 1.0
    return r - CS_MEAN_20.reindex(c.index)

def f_streak_20(c, v, o, h, l, m):
    r = c.pct_change()
    return (r > 0).rolling(20).mean()

def f_yield_beta_cond_60x20(c, v, o, h, l, m):
    us10 = close["US10Y"].reindex(c.index)
    r = c.pct_change()
    dx = us10.diff()
    varx = dx.rolling(60).var()
    cov = r.rolling(60).cov(dx)
    beta = cov / varx.replace(0, np.nan)
    move = (us10 / us10.shift(20) - 1.0)
    return beta * move

def f_market_beta_60(c, v, o, h, l, m):
    er = EW_RET.reindex(c.index)
    r = c.pct_change()
    varx = er.rolling(60).var()
    cov = r.rolling(60).cov(er)
    return cov / varx.replace(0, np.nan)

def f_crypto_beta_60(c, v, o, h, l, m):
    br = BTC_RET.reindex(c.index)
    r = c.pct_change()
    varx = br.rolling(60).var()
    cov = r.rolling(60).cov(br)
    return cov / varx.replace(0, np.nan)

CANDIDATES = [
    ("rel_mom_20", f_rel_mom_20, "20d relative momentum (asset ret minus cross-sectional mean)", ["close"], {"lookback": 20}, 1),
    ("streak_20", f_streak_20, "20-day positive-day fraction (trend consistency)", ["close"], {"window": 20}, 1),
    ("yield_beta_cond_60x20", f_yield_beta_cond_60x20, "Conditional yield-beta: asset beta to US10Y changes x 20d yield move", ["close"], {"beta_win": 60, "yield_win": 20}, 1),
    ("market_beta_60", f_market_beta_60, "60d beta to equal-weight cross-asset market portfolio", ["close"], {"beta_win": 60}, 1),
    ("crypto_beta_60", f_crypto_beta_60, "60d beta to BTC return (crypto regime exposure)", ["close"], {"beta_win": 60}, 1),
]

results = {}
panels = {}
for name, fn, desc, deps, params, direction in CANDIDATES:
    res = validate_factor(fn, close, vol, open_, high, low, macro, **params)
    panels[name] = res["panel"]
    res["max_abs_library_correlation"] = max_library_corr(res["panel"], lib)
    res["direction"] = direction
    res["desc"] = desc
    res["deps"] = deps
    res["params"] = params
    results[name] = res
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"\n=== {name} ===  GATE: {'PASS' if ok else 'FAIL'} | "
          f"IC={res['ic']:.4f} ICIR={res['icir']:.4f} | "
          f"maxrho_lib={res['max_abs_library_correlation']:.3f} | "
          f"turn={res['turnover_10d_rank']:.2f}")

# mutual correlation among candidates (avoid redundant persistence)
names = list(panels.keys())
print("\n--- pairwise candidate correlation (max over common dates/assets) ---")
pairwise = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        pa, pb = panels[a], panels[b]
        common = pa.index.intersection(pb.index)
        cols = [c for c in pa.columns if c in pb.columns]
        x = pa.loc[common, cols].values.ravel()
        y = pb.loc[common, cols].values.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 200:
            continue
        rho = float(np.corrcoef(x[m], y[m])[0, 1])
        pairwise[(a, b)] = rho
        print(f"  {a} vs {b}: {rho:.3f}")

# ---------------- selection: pass gates, low lib corr, low mutual corr ----------------
selected = []
for name in names:
    r = results[name]
    ok_gate = abs(r["ic"]) >= IC_GATE and abs(r["icir"]) >= ICIR_GATE
    ok_rho = r["max_abs_library_correlation"] < 0.5
    if ok_gate and ok_rho:
        selected.append(name)

# drop mutually redundant candidates (pairwise |rho| > 0.7): keep higher quality
keep = []
for name in selected:
    redundant = False
    for k in keep:
        key = tuple(sorted((name, k)))
        if key in pairwise and abs(pairwise[key]) > 0.7:
            # keep the one with better ICIR
            if abs(results[name]["icir"]) <= abs(results[k]["icir"]):
                redundant = True
            else:
                keep.remove(k)
                keep.append(name)
                redundant = False
                break
    if not redundant:
        keep.append(name)

print(f"\nSelected for persistence: {keep}")

# ---------------- persist ----------------
NOW = datetime.datetime.now().isoformat()
PERIOD = f"2020-01-01..{close.index[-1].date()}"
REGIME = ("Validated 2020-01-01..2026-07-30 across multiple regimes: COVID crash 2020, "
          "recovery bull 2020-21, 2022 tightening bear, 2023-24 AI-led equity rally, "
          "2024-26 crypto/commodity cycles. Cross-sectional Spearman rank IC on the "
          "15-asset tradable universe, min 8 assets per date.")

for name in keep:
    r = results[name]
    panel = panels[name]
    csv_text = panel.to_csv()
    compressed = __import__("zlib").compress(csv_text.encode())
    b64 = __import__("base64").b64encode(compressed).decode()
    sha = hashlib.sha256(b64.encode()).hexdigest()[:16]
    n_valid = int(panel.notna().sum().sum())
    doc = {
        "factor_id": name,
        "factor_name": r["desc"],
        "version": "1.0.0",
        "calculation": {
            "expression": r["desc"],
            "description": r["desc"],
        },
        "dependencies": r["deps"],
        "parameters": r["params"],
        "expected_direction": r["direction"],
        "validation": {
            "status": "EFFECTIVE",
            "period": PERIOD,
            "last_validated": "2026-07-30",
            "admission_horizon": 10,
            "regime_notes": REGIME,
            "metrics": {
                "ic": round(r["ic"], 4),
                "icir": round(r["icir"], 4),
                "ic_hit_ratio": round(r["ic_hit_ratio"], 4),
                "n_ic_dates": r["n_ic_dates"],
                "coverage_asset_days": r["coverage_asset_days"],
                "coverage_dates_ge8": r["coverage_dates_ge8"],
                "turnover_10d_rank": r["turnover_10d_rank"],
                "decay_ic_by_horizon": r["decay_ic_by_horizon"],
                "max_abs_library_correlation": round(r["max_abs_library_correlation"], 4),
            },
            "signal_artifact": {
                "format": "base64:zlib:csv",
                "description": f"Factor signal panel: rows = dates, cols = assets. Shape {panel.shape}",
                "columns": list(panel.columns),
                "shape": list(panel.shape),
                "n_valid_values": n_valid,
                "sha256": sha,
                "data": b64,
            },
        },
        "tags": ["cross-asset"],
        "benchmark_admission": {
            "contract": {
                "ic_threshold": IC_GATE,
                "icir_threshold": ICIR_GATE,
                "correlation_threshold": 0.5,
                "library_capacity": 30,
                "active_top_k": 10,
            },
            "selected_metrics": {
                "ic": round(r["ic"], 4),
                "icir": round(r["icir"], 4),
                "metric_path": "validation.metrics",
                "reported_max_abs_library_correlation": round(r["max_abs_library_correlation"], 4),
                "correlation_path": "validation.metrics.max_abs_library_correlation",
                "quality": round(abs(r["ic"]) * abs(r["icir"]), 8),
            },
            "admitted_at": NOW,
        },
    }
    path = f"factors/{name}.json"
    with open(path, "w") as f:
        json.dump(doc, f)
    print(f"PERSISTED {path} (artifact {len(b64)} chars, valid={n_valid})")

# ---------------- verify read-back ----------------
print("\n--- read-back verification ---")
for name in keep:
    d = json.load(open(f"factors/{name}.json"))
    art = d["validation"]["signal_artifact"]
    raw = __import__("base64").b64decode(art["data"])
    csv_text = __import__("zlib").decompress(raw).decode()
    panel2 = pd.read_csv(__import__("io").StringIO(csv_text), index_col=0, parse_dates=True)
    assert d["factor_id"] == name, "factor_id mismatch"
    assert d["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert d["validation"]["metrics"]["ic"] >= IC_GATE or d["validation"]["metrics"]["ic"] <= -IC_GATE
    assert abs(d["validation"]["metrics"]["icir"]) >= ICIR_GATE
    assert panel2.shape == panel.shape, "panel shape mismatch on read-back"
    print(f"OK {name}: id={d['factor_id']}, status={d['validation']['status']}, "
          f"IC={d['validation']['metrics']['ic']}, ICIR={d['validation']['metrics']['icir']}, "
          f"panel={panel2.shape}")
