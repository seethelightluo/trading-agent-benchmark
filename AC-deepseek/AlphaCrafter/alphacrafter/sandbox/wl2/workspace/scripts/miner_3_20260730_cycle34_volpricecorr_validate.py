"""miner_3 2026-07-30 cycle 34: validate vol_price_corr_60 (cycle-33 gate passer) for persistence.

Cycle 33 flagged vol_price_corr_60 (corr(|ret|, volume) over 60d) as pass=True:
  ic=0.0533 icir=0.1153 max_abs_lib_corr=0.142 (vs 3-factor active lib)
But it had low coverage (0.44 asset-days, 0.62 dates_ge8) and no regime check.
This script:
  - recomputes the signal (same construction as cycle 33)
  - full-metrics validation vs the FULL effective library (all signal artifacts)
  - regime breakdown (2020-21 / 2022 / 2023-24 / 2025-26 / last 250d)
  - window sensitivity (40/60/80) to ensure robustness, not overfit
  - coverage diagnostics by asset
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor, report,
                         VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}


def load_ohlc():
    out = {}
    for a in TRADABLES:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date")
        df = df.set_index("date")
        out[a] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out


OHLC = load_ohlc()


def vol_price_corr_panel(w=60, minp=30):
    """Rolling corr(volume, |daily return|) per asset on own calendar."""
    out = {}
    for a in TRADABLES:
        df = OHLC[a].dropna()
        ar = df["close"].pct_change().abs()
        c = ar.rolling(w, min_periods=minp).corr(df["volume"])
        out[a] = c.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


# --- full library: all effective factors with signal artifacts ---
lib = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20",
            "gain_loss_20", "intraday_drift_20", "usdjpy_beta_cond_120x60",
            "mom_10d_skip5", "mom_120d_skip5", "vol_of_vol20x60",
            "vix_beta_cond_60x20"]:
    npy = f"factors/{fid}.signal.npy"
    try:
        arr = np.load(npy)
        lib[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)
        continue
    except Exception:
        pass
    # recompute legacy signals (mirrors load_library_signals)
    close = panel
    if fid == "mom_10d_skip5":
        lib[fid] = per_asset(close, lambda s: s.shift(5) / s.shift(15) - 1.0)
    elif fid == "mom_120d_skip5":
        lib[fid] = per_asset(close, lambda s: s.shift(5) / s.shift(125) - 1.0)
    elif fid == "vol_of_vol20x60":
        lib[fid] = per_asset(close, lambda s: s.pct_change().rolling(20).std().rolling(60).std())
    elif fid == "vix_beta_cond_60x20":
        vix = macro_series("VIX").pct_change()
        parts = {}
        for a in close.columns:
            s = close[a].dropna()
            ar = s.pct_change()
            df = pd.concat([ar.rename("a"), vix.reindex(ar.index).rename("v")], axis=1).dropna()
            b = df["a"].rolling(60).cov(df["v"]) / df["v"].rolling(60).var()
            parts[a] = b.reindex(panel.index)
        beta_panel = pd.DataFrame(parts, index=panel.index)
        vix_close = macro_series("VIX")
        vix_20 = vix_close / vix_close.shift(20) - 1.0
        lib[fid] = -beta_panel.mul(vix_20.reindex(beta_panel.index), axis=0)
    else:
        print(f"!! no artifact for {fid}")
print(f"library: {sorted(lib)}")

# --- window sensitivity + validation ---
results = {}
for w, minp in [(40, 20), (60, 30), (80, 40)]:
    f = vol_price_corr_panel(w, minp)
    m = validate_factor(f, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=lib, fwd_cache=fwd_cache)
    p = report(f"vol_price_corr_{w}", m)
    print(f"=== vol_price_corr_{w} pass={p} ===")
    print(f"    ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']}")
    print(f"    cov_asset={m['coverage_asset_days']} cov_dates_ge8={m['coverage_dates_ge8']}")
    print(f"    turnover={m['turnover_10_rank']} decay={m['decay_ic_by_horizon']}")
    print(f"    max_abs_lib_corr={m['max_abs_library_correlation']}")
    top = sorted(m['library_pairwise_corr'].items(), key=lambda kv: -abs(kv[1]))[:4]
    print(f"    top corr: {top}")

    ic_ser = compute_ic(f, fwd_cache[str(ADM_H)]).dropna()
    parts = []
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 30:
            sd = sub.std()
            parts.append(f"{r0[:4]}:ic={sub.mean():+.4f}/icir={(sub.mean()/sd if sd > 0 else 0):+.3f}/n={len(sub)}")
    last = ic_ser.iloc[-250:]
    if len(last) >= 30:
        sd = last.std()
        parts.append(f"last250:ic={last.mean():+.4f}/icir={(last.mean()/sd if sd > 0 else 0):+.3f}/n={len(last)}")
    print(f"    regime: " + " | ".join(parts))
    results[f"vol_price_corr_{w}"] = {"metrics": m, "pass": p}
    print()

# coverage by asset for the 60d version
f60 = vol_price_corr_panel(60, 30)
cov_by_asset = f60.notna().sum().sort_values()
print("coverage by asset (60d):")
print(cov_by_asset.to_string())
print(f"total dates: {len(panel)}")

json.dump(results, open("scripts/_miner3_cycle34_volpricecorr_results.json", "w"), indent=1, default=float)
print("DONE cycle34")
