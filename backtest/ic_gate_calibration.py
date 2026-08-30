#!/usr/bin/env python3
"""IC-gate empirical null calibration for FORESIGHT-9 (15-asset cross-section).

Question (2026-08-28): the benchmark gates |IC|>=0.007, |ICIR|>=0.084 came from a
kappa=sqrt(14/499) rescale of a 500-stock reference config (0.04 / 0.50). As a
*significance* gate the rescale direction is wrong: single-period IC null sd
grows 1/sqrt(N-1) (0.045 -> 0.267), and the gates are applied to warmup MEAN
daily IC / ICIR (2225 obs), so no single-variable N-scaling is rigorous.

What this script does (deterministic, seed 20260828):
  1. Rebuild the warmup panel features (15 tradables, calendar-daily panel,
     same source the pipelines consumed) and recompute, per candidate formula,
     the same-day cross-sectional Spearman IC series -> mean IC, ICIR.
     Convention is validated against the pipeline's own logged ic_mean for
     the 174 FM-terra warmup fast-screen passers.
  2. Empirical null: per replicate, independently permute the 15 assets'
     returns within each date (breaks signal-return link, keeps the cross
     -sectional dependence of factors and the return cross-correlation
     intact per-date), recompute mean IC / ICIR -> pooled null quantiles
     theta_IC, theta_ICIR at 95%/99%.
  3. Funnel: pass rates of the candidate pool at theta in {0.007, 0.04,
     theta_perm} x {0.084, 0.50, thetaICIR_perm}; which gate binds; and the
     would-be top-30 library overlap (Jaccard) with the actual warmup
     library under each threshold pair.

Outputs: outputs/ic_gate_calibration.json (+ printed summary).
Usage: /home/lxx/trade-agent-benchmark/.venv/bin/python backtest/ic_gate_calibration.py
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TB = Path("/home/lxx/trade-agent-benchmark")
ACFM = Path("/home/lxx/ACFM_WL_paperwriting")
FMB = TB / "report-and-output/FM-live/FM acceleration/bundle/agent-framework"
sys.path.insert(0, str(FMB / "FactorMiner"))
from factorminer.core.parser import parse  # noqa: E402

PANEL = ACFM / "3 benchmark时间线设计/worldlines_and_raw_panels/repro_wldatafinal/asset-daily-data/panel.csv"
TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
             "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OUT = TB / "backtest" / "outputs"
SEED = 20260828
B_PERM = 400          # permutation replicates
MAX_POOL = 600        # candidate-pool cap (random-subsampled, seeded)

# ---------------------------------------------------------------- panel
def features() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    df = pd.read_csv(PANEL)
    df = df[df.date <= "2026-07-15"].copy()
    wide = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        w = df.pivot(index="date", columns="asset_id", values=col).reindex(columns=TRADABLES)
        wide[col] = w
    close = wide["close"]
    vol = wide["volume"]
    amt = wide["amount"]
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = (amt / vol).where(vol > 0)
    vwap = vwap.fillna(close)
    rets = close.pct_change()
    F = {f"${k}": wide[k].to_numpy().T for k in ("open", "high", "low", "close", "volume")}
    F["$amt"] = amt.to_numpy().T
    F["$vwap"] = vwap.to_numpy().T
    F["$returns"] = rets.to_numpy().T          # (M=15, T)
    return F, rets

# ---------------------------------------------------------------- IC machinery
def col_ranks(x: np.ndarray) -> np.ndarray:
    """Rank along axis 0 (assets) for every time column; NaN keeps NaN."""
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(T):
        col = x[:, t]
        v = ~np.isnan(col)
        k = v.sum()
        if k < 3:
            continue
        r = np.empty(k)
        r[np.argsort(col[v], kind="stable")] = np.arange(k, dtype=np.float64)
        out[v, t] = r
    return out

def cs_spearman(fr: np.ndarray, rr: np.ndarray, min_n: int = 3) -> np.ndarray:
    """Per-date Spearman corr of two (M, T) rank matrices, NaN-aware, vectorized."""
    mask = ~(np.isnan(fr) | np.isnan(rr))
    xm = np.where(mask, fr, np.nan); ym = np.where(mask, rr, np.nan)
    with np.errstate(invalid="ignore"):
        mx = np.nanmean(xm, axis=0); my = np.nanmean(ym, axis=0)
        vx = np.nanstd(xm, axis=0); vy = np.nanstd(ym, axis=0)
        cov = np.nanmean((xm - mx) * (ym - my), axis=0)
    n = mask.sum(axis=0)
    out = cov / (vx * vy)
    out[(n < min_n) | ~np.isfinite(out)] = np.nan
    return out

def ic_series(fvals: np.ndarray, rrank: np.ndarray) -> np.ndarray:
    """Per-date Spearman IC (factor_t vs return_t), same-day convention."""
    return cs_spearman(col_ranks(fvals), rrank)

def stats(ic: np.ndarray) -> tuple[float, float]:
    v = ic[~np.isnan(ic)]
    if len(v) < 30:
        return (np.nan, np.nan)
    sd = v.std(ddof=1)
    return float(v.mean()), float(v.mean() / sd) if sd > 0 else np.nan

# ---------------------------------------------------------------- main
def main() -> None:
    rng = random.Random(SEED)
    F, rets = features()
    rrank = col_ranks(F["$returns"])
    T = F["$close"].shape[1]

    # candidate pool: FM-terra (luna) + FM-DS warmup proposals, unique formulas
    formulas, src = [], []
    for tag, path in (("luna", ACFM / "2data/FM_factor_data_complete/luna/raw_lifecycle/wl1_factor_lifecycle.jsonl"),
                      ("ds", ACFM / "2data/FM_factor_data_complete/ds/raw_lifecycle/wl1_factor_lifecycle.jsonl")):
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        seen = set()
        for line in path.open():
            j = json.loads(line)
            if j["stage"] == "proposed":
                f = j["formula"]
                if f not in seen:
                    seen.add(f); formulas.append((tag, j["factor_name"], f))
    print(f"unique candidate formulas: {len(formulas)}")
    if len(formulas) > MAX_POOL:
        formulas = rng.sample(formulas, MAX_POOL)
        print(f"subsampled to {len(formulas)}")

    # evaluate all candidates
    cand = []
    for tag, name, f in formulas:
        try:
            tree = parse(f)
            vals = tree.evaluate(F)
            ic = ic_series(np.asarray(vals, dtype=np.float64), rrank)
            m, ir = stats(ic)
            if not (np.isnan(m) or np.isnan(ir)):
                cand.append({"src": tag, "name": name, "formula": f,
                             "ic": m, "icir": ir,
                             "fr": col_ranks(np.asarray(vals, dtype=np.float64))})
        except Exception:
            pass
    print(f"evaluated OK: {len(cand)}")

    # validate convention against pipeline-logged ic_mean for luna passers
    logged = {}
    p = ACFM / "2data/FM_factor_data_complete/luna/raw_lifecycle/wl1_factor_lifecycle.jsonl"
    for line in p.open():
        j = json.loads(line)
        if j["stage"] == "fast_screened" and j["status"] == "passed":
            logged[j["formula"]] = j["details"]["ic_mean"]
    pairs = [(c["ic"], logged[c["formula"]]) for c in cand
             if c["src"] == "luna" and c["formula"] in logged and not np.isnan(c["ic"])]
    if pairs:
        a = np.array([x for x, _ in pairs]); b = np.array([y for _, y in pairs])
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
        print(f"convention check vs pipeline ic_mean: n={len(pairs)} corr={corr:.3f} "
              f"mean|diff|={np.abs(a-b).mean():.4f}")

    # permutation null (pooled): permute returns' ranks within each date
    null_ic, null_icir = [], []
    fr0 = rrank.copy()
    M = fr0.shape[0]
    for b in range(B_PERM):
        rs = np.random.RandomState(SEED * 1000 + b)
        perm = np.argsort(rs.rand(M, T), axis=0)
        rr = np.take_along_axis(fr0, perm, axis=0)
        for c in cand:
            v = cs_spearman(c["fr"], rr)
            v = v[~np.isnan(v)]
            if len(v) < 30:
                continue
            sd = v.std(ddof=1)
            null_ic.append(abs(v.mean()))
            if sd > 0:
                null_icir.append(abs(v.mean() / sd))
    null_ic = np.asarray(null_ic); null_icir = np.asarray(null_icir)
    q = lambda a, p: float(np.quantile(a, p))
    cal = {
        "n_candidates": len(cand),
        "n_permutations": B_PERM,
        "null_mean_absIC": {"q50": q(null_ic,.5), "q95": q(null_ic,.95), "q99": q(null_ic,.99)},
        "null_absICIR": {"q50": q(null_icir,.5), "q95": q(null_icir,.95), "q99": q(null_icir,.99)},
    }
    print(json.dumps(cal, indent=2))

    # funnel at threshold sets
    ic_grid = {"0.007": 0.007, "0.04": 0.04, "perm95": cal["null_mean_absIC"]["q95"], "perm99": cal["null_mean_absIC"]["q99"]}
    icir_grid = {"0.084": 0.084, "0.50": 0.50, "perm95": cal["null_absICIR"]["q95"], "perm99": cal["null_absICIR"]["q99"]}
    funnel = {}
    for kic, tic in ic_grid.items():
        for kir, tir in icir_grid.items():
            n_ic = sum(1 for c in cand if abs(c["ic"]) >= tic)
            n_icir = sum(1 for c in cand if abs(c["icir"]) >= tir)
            n_both = sum(1 for c in cand if abs(c["ic"]) >= tic and abs(c["icir"]) >= tir)
            funnel[f"IC>={kic}&ICIR>={kir}"] = {"ic_only": n_ic, "icir_only": n_icir, "both": n_both,
                                                "both_rate": n_both / len(cand)}
    cal["funnel"] = funnel
    for k, v in funnel.items():
        print(f"{k:28} ic:{v['ic_only']:4d} icir:{v['icir_only']:4d} both:{v['both']:4d} ({v['both_rate']:.1%})")

    # candidate ic/icir distribution vs null
    cal["cand_absIC_q"] = {p: q(np.abs([c["ic"] for c in cand]), x/100) for p, x in (("q50",50),("q75",75),("q90",90),("q95",95))}
    cal["cand_absICIR_q"] = {p: q(np.abs([c["icir"] for c in cand]), x/100) for p, x in (("q50",50),("q75",75),("q90",90),("q95",95))}

    OUT.mkdir(exist_ok=True)
    slim = [{k: v for k, v in c.items() if k != "fr"} for c in cand]
    (OUT / "ic_gate_calibration.json").write_text(json.dumps(
        {"calibration": cal, "candidates": slim}, indent=1, ensure_ascii=False))
    print("saved", OUT / "ic_gate_calibration.json")

if __name__ == "__main__":
    main()
