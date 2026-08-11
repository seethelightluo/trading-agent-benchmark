"""Shared validation harness for miner_3 (2026-07-30 cycle).

Loads the 15-asset tradable universe + macro observation series from
../persistent/, computes factor values on per-asset series (shift semantics on
each asset's own calendar), computes rank IC / ICIR / turnover / coverage /
decay, and computes max_abs_library_correlation vs the persisted factor library
signal artifacts.

Admission gates (shared, 15-instrument universe):
  abs(IC) >= 0.0070 and abs(ICIR) >= 0.0840 at admission horizon 10.
"""
from __future__ import annotations
import base64, io, json, zlib
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("../persistent")
STOCK_DIR = DATA_DIR / "stock_data"
IDX_DIR = DATA_DIR / "index_data"
FACTOR_DIR = Path("factors")

ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

# Validation window (warm-up interval used by the existing library).
VALID_START = "2020-01-01"
VALID_END = "2026-07-15"      # factor-date cap; forward returns use data through 2026-07-29
VISIBLE_END = "2026-07-29"    # latest visible data (do not use anything after)


def load_closes():
    closes = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END]
        closes[a] = df.set_index("date")["close"].astype(float)
    return closes


def load_macro():
    out = {}
    for m in MACRO:
        f = IDX_DIR / f"{m}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END]
        out[m] = df.set_index("date")["close"].astype(float)
    return out


def weekday_grid(closes, start=VALID_START, end=VALID_END):
    dates = sorted({d for s in closes.values() for d in s.index})
    dates = [d for d in dates if start <= d.strftime("%Y-%m-%d") <= end and d.weekday() < 5]
    return pd.DatetimeIndex(dates)


def to_frame(closes, values):
    """Align per-asset factor series onto the common weekday grid -> DataFrame."""
    dates = weekday_grid(closes)
    df = pd.DataFrame(index=dates, columns=ASSETS, dtype=float)
    for a, s in values.items():
        if a in df.columns:
            df[a] = s.reindex(dates)
    return df


def forward_returns(closes, h):
    """Forward h-trading-day return per asset (per-asset own calendar shift)."""
    out = {}
    for a, s in closes.items():
        out[a] = s.shift(-h) / s - 1.0
    return out


def rank_ic(factor_frame, ret_frame, min_valid=8):
    """Daily cross-sectional Spearman IC between factor values and forward returns."""
    ics, dates = [], []
    for t in factor_frame.index:
        f = factor_frame.loc[t]
        r = ret_frame.loc[t]
        pair = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
        if len(pair) < min_valid:
            continue
        ic = pair["f"].rank().corr(pair["r"].rank())
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(t)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")


def turnover_rank10(factor_frame):
    """Mean abs change in cross-sectional rank percentile between 10-trading-day-spaced dates."""
    dates = factor_frame.index[::10]
    chg = []
    for i in range(1, len(dates)):
        a = factor_frame.loc[dates[i - 1]].rank(pct=True)
        b = factor_frame.loc[dates[i]].rank(pct=True)
        pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        if len(pair) >= 8:
            chg.append(float((pair["a"] - pair["b"]).abs().mean()))
    return float(np.mean(chg)) if chg else float("nan")


def decay_profile(closes, factor_frame, horizons=(1, 2, 3, 5, 10, 20)):
    prof = {}
    for h in horizons:
        rets = forward_returns(closes, h)
        ret_frame = pd.DataFrame({a: rets[a].reindex(factor_frame.index) for a in factor_frame.columns})
        ic = rank_ic(factor_frame, ret_frame)
        prof[str(h)] = round(float(ic.mean()), 4) if len(ic) else float("nan")
    return prof


def library_correlation(factor_frame):
    """Max abs Pearson correlation of candidate signal panel vs persisted library artifacts
    (restricted to the overlapping validation window)."""
    max_r = 0.0
    detail = {}
    for f in sorted(FACTOR_DIR.glob("*.json")):
        if f.name == "factor_ensemble.json":
            continue
        try:
            d = json.load(open(f))
            art = d.get("validation", {}).get("signal_artifact")
            if not art or "data" not in art:
                continue
            csv = zlib.decompress(base64.b64decode(art["data"])).decode()
            lib = pd.read_csv(io.StringIO(csv), index_col=0, parse_dates=True)
        except Exception as e:
            print(f"  [lib_corr] skip {f.name}: {e}")
            continue
        common_dates = factor_frame.index.intersection(lib.index)
        if len(common_dates) < 120:
            continue
        a = factor_frame.loc[common_dates]
        b = lib.loc[common_dates, factor_frame.columns]
        pair = pd.concat([a.stack().rename("x"), b.stack().rename("y")], axis=1).dropna()
        if len(pair) < 500:
            continue
        r = float(pair["x"].corr(pair["y"]))
        detail[f.name] = round(r, 4)
        max_r = max(max_r, abs(r))
    return max_r, detail


def evaluate(closes, values, label, horizon=10, verbose=True):
    """Full validation pipeline for one candidate factor."""
    frame = to_frame(closes, values)
    rets = forward_returns(closes, horizon)
    ret_frame = pd.DataFrame({a: rets[a].reindex(frame.index) for a in frame.columns})
    ic = rank_ic(frame, ret_frame)
    n_ic = len(ic)
    ic_mean = float(ic.mean()) if n_ic else float("nan")
    ic_std = float(ic.std(ddof=1)) if n_ic > 2 else float("nan")
    icir = ic_mean / ic_std if ic_std and np.isfinite(ic_std) else float("nan")
    hit = float((np.sign(ic) == np.sign(ic_mean)).mean()) if n_ic and ic_mean != 0 else float("nan")
    cov_asset_days = float(frame.notna().sum().sum() / (len(frame) * len(frame.columns)))
    cov_dates_ge8 = float((frame.notna().sum(axis=1) >= 8).mean())
    to = turnover_rank10(frame)
    decay = decay_profile(closes, frame)
    max_r, lib_detail = library_correlation(frame)

    passed = abs(ic_mean) >= 0.0070 and abs(icir) >= 0.0840
    if verbose:
        print(f"=== {label} ===")
        print(f"  factor dates: {len(frame)}  assets: {len(frame.columns)}")
        print(f"  n_ic_dates: {n_ic}  (dates with >=8 valid)")
        print(f"  IC (h={horizon}): {ic_mean:.4f}  ICIR: {icir:.4f}  hit: {hit:.3f}")
        print(f"  coverage_asset_days: {cov_asset_days:.3f}  coverage_dates_ge8: {cov_dates_ge8:.3f}")
        print(f"  turnover_10d_rank: {to:.3f}")
        print(f"  decay_ic_by_horizon: {decay}")
        print(f"  max_abs_library_correlation: {max_r:.4f}  detail: {lib_detail}")
        print(f"  GATE (absIC>=0.0070 & absICIR>=0.0840): {'PASS' if passed else 'FAIL'}")
    return {
        "label": label, "ic": ic_mean, "icir": icir, "hit": hit, "n_ic_dates": n_ic,
        "coverage_asset_days": cov_asset_days, "coverage_dates_ge8": cov_dates_ge8,
        "turnover_10d_rank": to, "decay": decay, "max_abs_library_correlation": max_r,
        "passed": passed, "frame": frame, "ic_series": ic,
    }


if __name__ == "__main__":
    closes = load_closes()
    print("assets loaded:", len(closes), "macro loaded:", len(load_macro()))
    # Calibration: reproduce mom_10d_skip5
    vals = {a: s.shift(5) / s.shift(15) - 1.0 for a, s in closes.items()}
    evaluate(closes, vals, "CALIB mom_10d_skip5 (expect IC~0.041 ICIR~0.118)")
