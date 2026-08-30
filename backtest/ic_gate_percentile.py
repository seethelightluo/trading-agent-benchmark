#!/usr/bin/env python3
"""Realized IC/ICIR admission boundary as a selectivity percentile.

Idea (2026-08-28): instead of arguing about N-scaling of the nominal gates
(|IC|>=0.007, |ICIR|>=0.084), measure what actually happened:
  - admitted = the four warmup libraries actually admitted at the boundary;
  - boundary = min |IC| (and min |ICIR|) among admitted factors;
  - percentile = where that boundary sits in the warmup candidate pool's
    |IC| (|ICIR|) distribution => "the realized gate keeps the top q%".

AC candidate pool: all mined warmup factor JSONs (accepted + rejected),
each carrying pipeline-computed validation.metrics.
FM candidate pool: unique proposed formulas from the warmup mining stream,
recomputed on the same warmup panel with the official parser/evaluator
(same-day cross-sectional Spearman IC series; convention validated against
the pipeline's logged ic_mean for fast-screen passers).

Usage: uv run (picturegenerate1 env: numpy/pandas/scipy/matplotlib).
Outputs: outputs/ic_gate_percentile.json + stdout summary.
"""
from __future__ import annotations

import glob
import json
import math
import random
import sys
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
POOL_CAP = 600

def col_ranks(x):
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(T):
        col = x[:, t]; v = ~np.isnan(col); k = v.sum()
        if k < 3: continue
        r = np.empty(k); r[np.argsort(col[v], kind="stable")] = np.arange(k, dtype=float)
        out[v, t] = r
    return out

def cs_spearman(fr, rr, min_n=3):
    mask = ~(np.isnan(fr) | np.isnan(rr))
    xm = np.where(mask, fr, np.nan); ym = np.where(mask, rr, np.nan)
    with np.errstate(invalid="ignore"):
        mx = np.nanmean(xm, axis=0); my = np.nanmean(ym, axis=0)
        vx = np.nanstd(xm, axis=0); vy = np.nanstd(ym, axis=0)
        cov = np.nanmean((xm - mx) * (ym - my), axis=0)
    out = cov / (vx * vy)
    out[(mask.sum(axis=0) < min_n) | ~np.isfinite(out)] = np.nan
    return out

def features():
    df = pd.read_csv(PANEL); df = df[df.date <= "2026-07-15"].copy()
    w = {c: df.pivot(index="date", columns="asset_id", values=c).reindex(columns=TRADABLES)
         for c in ("open","high","low","close","volume","amount")}
    vwap = (w["amount"]/w["volume"]).where(w["volume"]>0).fillna(w["close"])
    F = {f"${k}": v.to_numpy().T for k, v in w.items()}
    F["$vwap"] = vwap.to_numpy().T
    F["$returns"] = w["close"].pct_change().to_numpy().T
    return F

def ac_metric(j, *keys):
    m = (j.get("validation") or {}).get("metrics") or {}
    for k in keys:
        v = m.get(k)
        if isinstance(v, (int, float)): return float(v)
    return float("nan")

def main():
    rng = random.Random(SEED)
    F = features()
    # next-day convention: IC_t = corr(factor_t, r_{t+1})  (predictive, matches pipeline)
    fwd = np.full_like(F["$returns"], np.nan)
    fwd[:, :-1] = F["$returns"][:, 1:]
    rrank = col_ranks(fwd)

    # ---------- admitted sets ----------
    admitted = {}
    for tag, csvp in (("fm-terra", "luna"), ("fm-ds", "ds")):
        p = ACFM / "2data/FM_factor_data_complete" / csvp / f"warmup_library_{csvp}.csv"
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig"))) if (csv := __import__("csv")) else []
        admitted[tag] = [(r["name"], abs(float(r["ic_mean"])), abs(float(r["icir"])))
                         for r in rows if r.get("ic_mean") and r.get("icir")]
    for tag, d in (("ac-terra", TB/"agent-framework/AlphaCrafter/alphacrafter/sandbox/ws1/workspace/factors"),
                   ("ac-ds", TB/"AC-deepseek/AlphaCrafter/alphacrafter/sandbox/ws1/workspace/factors")):
        rows = []
        for f in sorted(glob.glob(str(d/"*.json"))):
            if f.endswith(("factor_ensemble.json",".bak")): continue
            j = json.load(open(f))
            ic = ac_metric(j,"mean_daily_paper_ic","ic_mean_daily","mean_daily_ic","ic")
            ir = ac_metric(j,"daily_paper_icir","icir_daily","icir")
            if not math.isnan(ic): rows.append((j["factor_id"], abs(ic), abs(ir)))
        admitted[tag] = rows

    # ---------- candidate pools ----------
    # AC: all mined JSONs incl. rejected
    ac_pool = {}
    for tag, d in (("ac-terra", TB/"agent-framework/AlphaCrafter/alphacrafter/sandbox/ws1"),
                   ("ac-ds", TB/"AC-deepseek/AlphaCrafter/alphacrafter/sandbox/ws1")):
        ics, irs = [], []
        for f in glob.glob(str(d/"workspace/factors/*.json")) + glob.glob(str(d/"workspace/factors/rejected/*.json")):
            if f.endswith(("factor_ensemble.json",".bak",".reason.json")): continue
            j = json.load(open(f))
            ic = ac_metric(j,"mean_daily_paper_ic","ic_mean_daily","mean_daily_ic","ic")
            ir = ac_metric(j,"daily_paper_icir","icir_daily","icir")
            if not math.isnan(ic): ics.append(abs(ic)); irs.append(abs(ir))
        ac_pool[tag] = (np.array(ics), np.array(irs))

    # FM: recompute on warmup panel -- WARMUP-ONLY pool (iteration <= 200).
    # (ds/raw_lifecycle holds the ONLINE stream only: iterations all >200;
    #  the DS warmup proposals are not archived, so the warmup candidate
    #  distribution comes from the terra stream -- same mining machinery.)
    formulas, src = [], []
    for tag, tagd in (("fm-terra","luna"),):
        p = ACFM/"2data/FM_factor_data_complete"/tagd/"raw_lifecycle/wl1_factor_lifecycle.jsonl"
        seen = set()
        for line in p.open():
            j = json.loads(line)
            if j["stage"] == "proposed" and j["iteration"] <= 200 and j["formula"] not in seen:
                seen.add(j["formula"]); formulas.append((tag, j["formula"]))
    all_f = [(t, f) for t, f in formulas]
    print("unique FM warmup proposals:", len(all_f))
    sample = all_f if len(all_f) <= POOL_CAP else rng.sample(all_f, POOL_CAP)
    fm_pool = {"fm-terra": ([], []), "fm-ds": ([], [])}
    n_ok = 0
    for tag, f in sample:
        try:
            vals = np.asarray(parse(f).evaluate(F), dtype=np.float64)
            ic = cs_spearman(col_ranks(vals), rrank)
            v = ic[~np.isnan(ic)]
            if len(v) < 30: continue
            sd = v.std(ddof=1)
            if sd <= 0: continue
            fm_pool[tag][0].append(abs(v.mean())); fm_pool[tag][1].append(abs(v.mean()/sd)); n_ok += 1
        except Exception:
            pass
    print("FM recomputed OK:", n_ok)

    # convention check vs pipeline-logged ic_mean (luna fast-screen passers)
    logged = {}
    lp = ACFM/"2data/FM_factor_data_complete/luna/raw_lifecycle/wl1_factor_lifecycle.jsonl"
    for line in lp.open():
        j = json.loads(line)
        if j["stage"] == "fast_screened" and j["status"] == "passed":
            logged[j["formula"]] = j["details"]["ic_mean"]
    pairs = []
    for tag, f in sample:
        if tag == "fm-terra" and f in logged:
            pass
    cache = {}
    for tag, f in sample:
        if tag == "fm-terra" and f in logged:
            try:
                vals = np.asarray(parse(f).evaluate(F), dtype=np.float64)
                ic = cs_spearman(col_ranks(vals), rrank)
                v = ic[~np.isnan(ic)]
                if len(v) >= 30 and v.std(ddof=1) > 0:
                    cache[f] = v.mean()
            except Exception:
                pass
    a = np.array([cache[f] for f in logged if f in cache])
    b = np.array([logged[f] for f in logged if f in cache])
    print(f"convention check (next-day) vs pipeline ic_mean: n={len(a)} "
          f"corr={np.corrcoef(a,b)[0,1]:.3f} mean|diff|={np.abs(a-b).mean():.4f}")

    # ---------- report ----------
    def pct(pool, x):  # fraction of pool strictly below x
        return float((pool < x).mean())
    res = {"admitted": {}, "pools": {}, "boundary": {}}
    pools = {"ac-terra": ac_pool["ac-terra"], "ac-ds": ac_pool["ac-ds"],
             "fm-terra": tuple(map(np.array, fm_pool["fm-terra"])),
             "fm-ds": tuple(map(np.array, fm_pool["fm-ds"]))}
    all_ic = np.concatenate([p[0] for p in pools.values() if len(p[0])])
    all_ir = np.concatenate([p[1] for p in pools.values() if len(p[1])])
    # fm-ds has no warmup candidate archive -> use the terra warmup pool for its boundary too
    if not len(pools["fm-ds"][0]):
        pools["fm-ds"] = pools["fm-terra"]
    for tag, rows in admitted.items():
        mic = min(r[1] for r in rows); mir = min(r[2] for r in rows)
        pic, pir = pools[tag][0], pools[tag][1]
        res["admitted"][tag] = {"n": len(rows), "min_absIC": mic, "min_absICIR": mir,
            "members": [(n, round(i,4), round(r,4)) for n,i,r in rows]}
        res["pools"][tag] = {"n": len(pic)}
        res["boundary"][tag] = {
            "ic_percentile_of_pool": pct(pic, mic) if len(pic) else None,
            "icir_percentile_of_pool": pct(pir, mir) if len(pir) else None,
            "ic_top_share": 1-pct(pic, mic) if len(pic) else None,
            "icir_top_share": 1-pct(pir, mir) if len(pir) else None}
    res["pooled"] = {
        "min_absIC_over_admitted": min(r[1] for rows in admitted.values() for r in rows),
        "min_absICIR_over_admitted": min(r[2] for rows in admitted.values() for r in rows),
        "pooled_ic_percentile": pct(all_ic, min(r[1] for rows in admitted.values() for r in rows)),
        "pooled_icir_percentile": pct(all_ir, min(r[2] for rows in admitted.values() for r in rows)),
        "nominal_gate_0.007_percentile": pct(all_ic, 0.007),
        "nominal_gate_0.084_percentile": pct(all_ir, 0.084)}
    OUT.mkdir(exist_ok=True)
    (OUT/"ic_gate_percentile.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "admitted"}, indent=1))
    for tag, rows in admitted.items():
        b = res["boundary"][tag]
        mi, mr = min(r[1] for r in rows), min(r[2] for r in rows)
        ps = f"top {b['ic_top_share']:.1%} of pool n={res['pools'][tag]['n']}" if b["ic_top_share"] else "no pool"
        pr = f"top {b['icir_top_share']:.1%}" if b["icir_top_share"] else "no pool"
        print(f"{tag}: n_adm={len(rows)} min|IC|={mi:.4f} ({ps}) min|ICIR|={mr:.4f} ({pr})")

if __name__ == "__main__":
    main()
