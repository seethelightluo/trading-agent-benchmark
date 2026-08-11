"""miner_2: Persist v7b PASS candidates with full validation + signal artifacts.

Candidates (h=10, VIS=2026-07-29, all passed |IC|>=0.007 and |ICIR|>=0.084):
  time_under_water_120, rsi_14, semi_down_ratio_20, ret_vol_ratio_20, skew_20_raw
Artifacts are base64(zlib(csv)) of the signal panel (dates x 15 assets) so the
deterministic post-miner gate can recompute pairwise rho from real signals.
"""
import sys, os, json, io, base64, zlib, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
LIB_DIR = "factors"
OUT_DIR = "factors"
TODAY = "2026-07-30"

close = closes_panel(VIS)
idx = close.index
fr = forward_returns(close, H)

# ---------------- library IC map by decoding artifacts ----------------
def decode_artifact(meta):
    a = meta.get("validation", {}).get("signal_artifact")
    if not a:
        return None
    dec = zlib.decompress(base64.b64decode(a["data"])).decode("utf-8")
    sig = pd.read_csv(io.StringIO(dec), index_col=0, parse_dates=True)
    return sig.reindex(columns=close.columns).reindex(close.index)


def build_lib_ic_map():
    lib = {}
    for fn in sorted(os.listdir(LIB_DIR)):
        if not fn.endswith(".json") or fn == "factor_ensemble.json":
            continue
        with open(os.path.join(LIB_DIR, fn)) as f:
            meta = json.load(f)
        sig = decode_artifact(meta)
        if sig is None:
            continue
        ic = ic_series(sig, fr, min_valid=8)
        if len(ic.dropna()) > 30:
            lib[meta["factor_id"]] = ic
    return lib


lib_ics = build_lib_ic_map()
print("library IC series decoded:", len(lib_ics))


def rho_vs_lib(my_ic, exclude=()):
    best, best_id = 0.0, None
    for fid, s in lib_ics.items():
        if fid in exclude:
            continue
        pair = pd.concat([my_ic.rename("a"), s.rename("b")], axis=1).dropna()
        if len(pair) < 30:
            continue
        r = pair["a"].corr(pair["b"])
        if np.isfinite(r) and abs(float(r)) > best:
            best, best_id = abs(float(r)), fid
    return round(best, 4), best_id


# ---------------- per-asset-calendar factor builders ----------------
def clean(s):
    return s.dropna()


def roll(s, fn):
    c = clean(s)
    if len(c) < 30:
        return pd.Series(np.nan, index=idx)
    return fn(c).reindex(idx)


vol_data, hl_data = {}, {}
for s in WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
    d2 = df.set_index("date")
    vol_data[s] = d2["volume"].astype(float).reindex(idx)
    hl_data[s] = d2[["high", "low"]].reindex(idx)


def f_time_under_water(c):
    # days since the last 120d rolling max
    def f(x):
        return len(x) - 1 - int(np.argmax(x))
    return c.rolling(120).apply(f, raw=True)


def f_rsi(c):
    delta = c.diff()
    up = delta.clip(lower=0.0).rolling(14).mean()
    dn = (-delta.clip(upper=0.0)).rolling(14).mean()
    return 100.0 - 100.0 / (1.0 + (up / dn.replace(0, np.nan)))


def f_semi(c):
    r = c.pct_change()
    d = (r.clip(upper=0.0) ** 2).rolling(20).mean().apply(np.sqrt)
    u = (r.clip(lower=0.0) ** 2).rolling(20).mean().apply(np.sqrt)
    return (d / u.replace(0, np.nan) - 1.0)


def f_rv(c):
    r = c.pct_change()
    return r.rolling(20).mean() / r.rolling(20).std().replace(0, np.nan)


def f_skew(c):
    return c.pct_change().rolling(20).skew()


BUILDERS = {
    "time_under_water_120": {
        "fn": f_time_under_water,
        "name": "Time under water 120d",
        "expr": "days_since_last_rolling_max(close, 120)",
        "desc": "Drawdown-duration timing factor: number of trading days since the asset last "
                "printed a new 120-day closing high. Long underwater streaks flag weak/oversold "
                "assets that keep underperforming over the next 10d (negative IC), a mean-"
                "reversion-averse drawdown-timing signal.",
        "direction": -1,
        "params": {"window": 120, "horizon": 10, "min_valid_assets": 8},
        "tags": ["drawdown", "timing", "trend", "cross-asset"],
    },
    "rsi_14": {
        "fn": f_rsi,
        "name": "Relative Strength Index 14d",
        "expr": "100 - 100/(1 + avg_gain_14/avg_loss_14)",
        "desc": "Classic 14-day RSI (simple-mean variant). In this cross-asset universe high RSI "
                "(strong recent momentum) continues to outperform over 10d; positive IC in the "
                "2020-22 and 2025-26 regimes, flat in 2023-24.",
        "direction": 1,
        "params": {"rsi_win": 14, "horizon": 10, "min_valid_assets": 8},
        "tags": ["momentum", "oscillator", "mean-reversion", "cross-asset"],
    },
    "semi_down_ratio_20": {
        "fn": f_semi,
        "name": "Downside/upside semi-vol ratio 20d",
        "expr": "sqrt(mean(min(pct_change,0)^2,20))/sqrt(mean(max(pct_change,0)^2,20)) - 1",
        "desc": "Crash-asymmetry gauge: ratio of downside to upside semi-deviation over 20d minus 1. "
                "Assets dominated by large negative moves (ratio > 0) underperform over 10d; "
                "negative IC stable across all three regimes.",
        "direction": -1,
        "params": {"win": 20, "horizon": 10, "min_valid_assets": 8},
        "tags": ["risk", "asymmetry", "crash-risk", "cross-asset"],
    },
    "ret_vol_ratio_20": {
        "fn": f_rv,
        "name": "Risk-adjusted trend 20d",
        "expr": "mean(pct_change,20)/std(pct_change,20)",
        "desc": "20-day return per unit of 20-day volatility (Sharpe-like trend). Positive IC with "
                "strong 2025-26 regime (ICIR 0.216); robust continuation signal.",
        "direction": 1,
        "params": {"win": 20, "horizon": 10, "min_valid_assets": 8},
        "tags": ["momentum", "risk-adjusted", "trend", "cross-asset"],
    },
    "skew_20_raw": {
        "fn": f_skew,
        "name": "Return skewness 20d raw",
        "expr": "skew(pct_change, 20)",
        "desc": "Raw 20-day return skewness. Positively-skewed assets (right-tail up-moves) "
                "outperform over 10d; positive IC in all three regimes (0.109/0.071/0.175 ICIR).",
        "direction": 1,
        "params": {"win": 20, "horizon": 10, "min_valid_assets": 8},
        "tags": ["skew", "momentum", "cross-asset"],
    },
}


def make_artifact(sig):
    buf = io.StringIO()
    sig.to_csv(buf, date_format="%Y-%m-%d")
    csv_str = buf.getvalue()
    data_b64 = base64.b64encode(zlib.compress(csv_str.encode("utf-8"))).decode("ascii")
    sha = hashlib.sha256(csv_str.encode("utf-8")).hexdigest()
    return {
        "format": "base64:zlib:csv",
        "description": "Factor signal panel: rows = dates (YYYY-MM-DD), cols = 15 watchlist symbols. "
                       "Recover with zlib.decompress(base64.b64decode(data)).decode() -> "
                       "pandas.read_csv(StringIO).",
        "columns": list(sig.columns),
        "shape": list(sig.shape),
        "n_valid_values": int(sig.notna().sum().sum()),
        "sha256": sha,
        "data": data_b64,
    }


# ---------------- compute + persist ----------------
results = {}
for fid, spec in BUILDERS.items():
    sig = pd.DataFrame({s: roll(close[s], spec["fn"]) for s in WATCH}, index=idx)
    sig = sig.reindex(columns=close.columns).reindex(close.index)
    ic = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic, sig, fr, close, h=H)
    if m is None:
        print(f"{fid:22s} INSUFFICIENT dates ({len(ic.dropna())}) - skip")
        continue
    m["regime"] = regime_split(ic)
    rho, rho_id = rho_vs_lib(ic, exclude=set(BUILDERS.keys()))
    m["max_abs_library_correlation"] = rho
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    if not gate:
        print(f"{fid:22s} FAILS gate ic={m['ic']:+.4f} icir={m['icir']:+.4f} - skip")
        continue
    results[fid] = m
    artifact = make_artifact(sig)
    meta = {
        "factor_id": fid,
        "factor_name": spec["name"],
        "version": "1.0.0",
        "calculation": {
            "expression": spec["expr"],
            "description": spec["desc"],
        },
        "dependencies": ["close"],
        "parameters": spec["params"],
        "expected_direction": spec["direction"],
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-29",
            "admission_horizon": H,
            "last_validated": TODAY,
            "regime_notes": (
                f"Validated on the 15-asset cross-asset universe; {m['n_ic_dates']} IC dates with "
                f">=8 valid instruments. Regime ICs: {json.dumps({k: v['ic'] for k, v in m['regime'].items()})}. "
                f"Max abs IC correlation with decodable library factors: {rho} (vs {rho_id})."
            ),
            "metrics": m,
            "signal_artifact": artifact,
        },
        "tags": spec["tags"],
    }
    fp = os.path.join(OUT_DIR, fid + ".json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"WROTE {fp}")
    print(f"  ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} turn={m.get('turnover_10d_rank')} "
          f"rho={rho} (vs {rho_id})")

print("\nSaved results summary ->", json.dumps({k: {"ic": v["ic"], "icir": v["icir"],
      "rho": v["max_abs_library_correlation"]} for k, v in results.items()}, indent=1))

# ---------------- verification pass ----------------
print("\n--- VERIFICATION (reload + artifact recovery) ---")
for fid in results:
    fp = os.path.join(OUT_DIR, fid + ".json")
    with open(fp, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["factor_id"] == fid
    assert meta["validation"]["status"] == "EFFECTIVE"
    sig2 = decode_artifact(meta)
    assert sig2 is not None and sig2.shape == (len(idx), 15)
    ic2 = ic_series(sig2, fr, min_valid=8)
    ic2m = float(ic2.dropna().mean())
    ok = abs(ic2m - results[fid]["ic"]) < 5e-4
    print(f"{fid:22s} reload OK  shape={sig2.shape} n_valid={int(sig2.notna().sum().sum())} "
          f"recovered_ic={ic2m:+.4f} match={ok}")
    assert ok
print("ALL VERIFIED")
