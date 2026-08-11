"""miner_2: Explore new factor families with full library-correlation check.

Library IC series are rebuilt by DECODING each persisted factor's signal
artifact (base64:zlib:csv), so rho is accurate vs the deterministic gate.

Candidates explored (one batch for screening; each persisted factor gets its
own focused validation + artifact):
  dd_60, dd_120            drawdown depth vs rolling max
  rev_5d                   short-term 5d reversal
  rsi_14                   classic RSI oscillator
  amihud_20                Amihud illiquidity (|ret|/volume)
  vol_trend_20x60          volume expansion ratio
  semi_down_ratio_20       downside/upside semi-vol asymmetry
  beta_ew_60               beta vs equal-weight cross-asset index
  corr_ew_60               correlation vs equal-weight index
  rng_pos_20               20d range position
  ret_vol_ratio_20         return/vol ratio (risk-adjusted trend)
  mom_60_skip5             60d momentum (skip 5) - redundancy reference
  us10y_beta_60            beta vs US10Y daily changes
"""
import sys, os, json, io, base64, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
LIB_DIR = "factors"

close = closes_panel(VIS)
ret = close.pct_change()
fr = forward_returns(close, H)

# ---- library IC map by decoding artifacts ----
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
        fid = meta["factor_id"]
        sig = decode_artifact(meta)
        if sig is None:
            # expression fallback for trend_r2_30_signed (no artifact yet)
            expr = meta.get("calculation", {}).get("expression", "")
            try:
                if "corr" in expr and "rank" not in expr:
                    continue  # skip unknowns
            except Exception:
                pass
            continue
        ic = ic_series(sig, fr, min_valid=8)
        if len(ic.dropna()) > 30:
            lib[fid] = ic
    return lib


lib_ics = build_lib_ic_map()
print("library IC series decoded:", {k: len(v.dropna()) for k, v in lib_ics.items()})


def rho_vs_lib(my_ic):
    best = 0.0
    for fid, s in lib_ics.items():
        pair = pd.concat([my_ic.rename("a"), s.rename("b")], axis=1).dropna()
        if len(pair) < 30:
            continue
        r = pair["a"].corr(pair["b"])
        if np.isfinite(r):
            best = max(best, abs(float(r)))
    return round(best, 4)


# ---- candidate builders ----
def build_candidates():
    idx = close.index
    n = len(idx)
    high = close  # fallback; real high/low used where available
    cands = {}

    rollmax60 = close.rolling(60).max()
    rollmax120 = close.rolling(120).max()
    cands["dd_60"] = close / rollmax60 - 1.0
    cands["dd_120"] = close / rollmax120 - 1.0
    cands["rev_5d"] = 1.0 - close / close.shift(5)
    cands["mom_60_skip5"] = close.shift(5) / close.shift(65) - 1.0

    # RSI(14)
    delta = close.diff()
    up = delta.clip(lower=0.0).rolling(14).mean()
    dn = (-delta.clip(upper=0.0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    cands["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    # Amihud illiquidity (uses volume; missing volume -> NaN)
    vol = close.copy()
    for s in close.columns:
        try:
            df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
            v = df.set_index("date")["volume"].astype(float).reindex(idx)
            vol[s] = v
        except Exception:
            vol[s] = np.nan
    illiq = (ret.abs() / vol).rolling(20).mean()
    cands["amihud_20"] = illiq

    # volume trend
    vol_trend = vol.rolling(20).mean() / vol.rolling(60).mean() - 1.0
    cands["vol_trend_20x60"] = vol_trend

    # semi-deviation asymmetry: downside semi-vol / upside semi-vol - 1
    neg = ret.clip(upper=0.0)
    pos = ret.clip(lower=0.0)
    down_sv = (neg ** 2).rolling(20).mean().apply(np.sqrt)
    up_sv = (pos ** 2).rolling(20).mean().apply(np.sqrt)
    cands["semi_down_ratio_20"] = down_sv / up_sv.replace(0, np.nan) - 1.0

    # equal-weight index
    ew = ret.mean(axis=1)
    ew_ret = ew.rename("ew")
    beta_ew = {}
    corr_ew = {}
    for a in close.columns:
        pair = pd.concat([ret[a].rename("a"), ew_ret], axis=1).dropna()
        cov = pair["a"].rolling(60).cov(pair["ew"])
        var = pair["ew"].rolling(60).var()
        beta_ew[a] = cov / var.replace(0, np.nan)
        corr_ew[a] = pair["a"].rolling(60).corr(pair["ew"])
    cands["beta_ew_60"] = pd.DataFrame(beta_ew, index=idx)
    cands["corr_ew_60"] = pd.DataFrame(corr_ew, index=idx)

    # 20d range position using high/low where present
    rp = {}
    for s in close.columns:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
        d2 = df.set_index("date").reindex(idx)
        if d2["high"].isna().all() or d2["low"].isna().all():
            rp[s] = np.nan
            continue
        hi = d2["high"].rolling(20).max()
        lo = d2["low"].rolling(20).min()
        rng = (hi - lo).replace(0, np.nan)
        rp[s] = (d2["close"] - lo) / rng
    cands["rng_pos_20"] = pd.DataFrame(rp, index=idx)

    # return/vol ratio (risk-adjusted trend)
    cands["ret_vol_ratio_20"] = ret.rolling(20).mean() / ret.rolling(20).std().replace(0, np.nan)

    # US10Y beta (rate sensitivity)
    us10 = pd.read_csv("../persistent/stock_data/US10Y.csv", parse_dates=["date"])
    us10r = us10.set_index("date")["close"].astype(float).reindex(idx).pct_change().rename("u")
    beta_us = {}
    for a in close.columns:
        pair = pd.concat([ret[a].rename("a"), us10r], axis=1).dropna()
        b = pair["a"].rolling(60).cov(pair["u"]) / pair["u"].rolling(60).var().replace(0, np.nan)
        beta_us[a] = b
    cands["us10y_beta_60"] = pd.DataFrame(beta_us, index=idx)

    return cands


cands = build_candidates()
print("\n--- CANDIDATE SCREEN (h=%d) ---" % H)
results = {}
for name, sig in cands.items():
    sig = sig.reindex(columns=close.columns).reindex(close.index)
    ic = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic, sig, fr, close, h=H)
    if m is None:
        print(f"{name:20s} insufficient dates ({len(ic.dropna())})")
        continue
    m["rho"] = rho_vs_lib(ic)
    m["regime"] = regime_split(ic)
    results[name] = m
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    flag = "PASS" if gate else "fail"
    print(f"{name:20s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} cov={m['coverage_asset_days']:.2f} rho={m['rho']:.3f} [{flag}]")
    print("     regime:", json.dumps(m["regime"]))

with open("scripts/miner2_20260730_explore_v7_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved results -> scripts/miner2_20260730_explore_v7_results.json")
