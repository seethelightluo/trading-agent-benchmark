#!/usr/bin/env python3
"""Recompute the Terra wl2/wl3 online NAV path that was never rebalanced.

Facts established by audit (2026-08-12):
- wl2/wl3 rebalance_history has only the 2026-07-30 initial allocation.
- Every decision had empty forecast_returns/factor_ids and skip_reason
  'missing_forecast_proposal', so the migration gate skipped all rebalances:
  holdings stayed at the initial allocation for ~4 sim years.
- The recorded NAV therefore reflects pure price drift, not factor rebalancing.
- Sandbox stock_data matches WL-data-final panels (1e-12), but the recorded
  NAV itself was produced with an earlier data vintage, so the true path must
  be recomputed on WL-data-final.

Replay design (documented in runAC.md):
- Anchor: the only verified executed state, rebalance_history[0]
  (2026-07-30, post-trade NAV, target weights).
- Proposals: recorded decision_history.proposed_target_weights (the trader's
  intended targets at each 10-trading-day block).
- Forecasts: deterministic signals from the current aligned strategy
  (factor_ensemble.json + factor-family proxies, 1-bar lag), so the
  migration gate can be applied retroactively.
- Gate (contract): gross_edge_bps > 3 * one_way_turnover (strict).
  gross_edge_bps = 10000 * sum((proposed-current) * forecast).
  one_way_turnover = 0.5 * sum(abs(proposed-current)).
  cost = NAV * turnover * 3 / 10000 on execution.
- Variants: 'gate' (contract) and 'all' (execute every proposal).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lxx/trade-agent-benchmark")
ALPHA = ROOT / "agent-framework/AlphaCrafter/alphacrafter"
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
          "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
COST_BPS = 3.0
DATA_DAYS = 170
MIN_ROWS = 140


def _num(text: str, key: str, default: int) -> int:
    m = re.search(key + r"[_ ]*(\d+)", text, re.I)
    return int(m.group(1)) if m else default


def factor_series(close: pd.Series, fid: str) -> pd.Series:
    """Deterministic 1-bar-lagged factor proxy, mirrored from strategy.py."""
    c = close.astype(float)
    r = c.pct_change()
    lr = np.log(c).diff()
    f = fid.lower()
    if "reversal" in f or "_rev_" in f:
        w = _num(f, "3d", 3) if "3d" in f else (_num(f, "1d", 1) if "1d" in f else 5)
        return -c.pct_change(w).shift(1)
    if "compression" in f or ("vol" in f and "comp" in f):
        s = _num(f, "10", 10)
        lw = _num(f, "60", 60)
        return -(lr.rolling(s).std() / lr.rolling(lw).std()).shift(1)
    if "breakout" in f or "distance" in f:
        w = 120 if "120" in f else 60
        return ((c / c.rolling(w).max() - 1.0)).shift(1)
    if "consistency" in f or "persistence" in f or "accel" in f:
        return ((c.pct_change(30)) * (r > 0).rolling(30).mean()).shift(1)
    if "breadth" in f or "asymmetry" in f or "quality" in f:
        return ((r > 0).rolling(30).mean() - 0.5).shift(1)
    if "volatility" in f or "volstate" in f or "_vol_" in f or "risk" in f:
        return (-r.rolling(20).std()).shift(1)
    if "beta" in f or "residual" in f:
        bench = c.pct_change().rolling(60).mean()
        return (r - bench).shift(1)
    if "leadlag" in f or "lead" in f:
        return (c.pct_change(5) - c.pct_change(5).rolling(5).mean()).shift(1)
    if "stress" in f:
        return (-(lr.rolling(5).std() / lr.rolling(20).std())).shift(1)
    if "dispersion" in f:
        return (-r.rolling(5).std()).shift(1)
    if "downside" in f:
        neg = r.copy()
        neg[neg > 0] = np.nan
        return (c.pct_change(20) / (neg.rolling(30).std() + 1e-9)).shift(1)
    return c.pct_change(20).shift(1)


def forecasts_at(px: pd.DataFrame, ensemble: list, visible_through: str) -> dict | None:
    """Cross-sectional score -> forecast map at a decision date."""
    scores = {a: 0.0 for a in ASSETS}
    used = 0
    for fid, w, direction in ensemble:
        vals = {}
        ok = 0
        for a in ASSETS:
            try:
                hist = px.loc[:visible_through, a].dropna().tail(DATA_DAYS)
                if len(hist) < MIN_ROWS:
                    vals[a] = None
                    continue
                s = factor_series(hist, fid)
                s = s.replace([np.inf, -np.inf], np.nan)
                v = float(s.iloc[-1])
                vals[a] = v if math.isfinite(v) else None
                if vals[a] is not None:
                    ok += 1
            except Exception:
                vals[a] = None
        if ok < 8:
            continue
        valid = sorted((float(v), a) for a, v in vals.items() if v is not None)
        rk = {a: 0.5 for a in ASSETS}
        n = len(valid)
        for i, (_, a) in enumerate(valid):
            rk[a] = i / max(1, n - 1)
        for a in ASSETS:
            scores[a] += w * (rk[a] if direction > 0 else 1.0 - rk[a])
        used += 1
    if used < 5:
        return {a: 0.0 for a in ASSETS}, used
    vals = [scores[a] for a in ASSETS]
    mean = float(np.mean(vals))
    half = max(1e-9, (max(vals) - min(vals)) / 2.0)
    f = {}
    for a in ASSETS:
        z = (scores[a] - mean) / half
        f[a] = float(np.clip(0.04 * z, -0.05, 0.05))
    return f, used


def load_worldline(wl: int) -> dict:
    sandbox = ALPHA / f"sandbox/wl{wl}"
    ac = json.load(open(sandbox / "persistent/account.json"))
    ens = json.load(open(sandbox / "workspace/factor_ensemble.json"))
    factors = [(x["factor_id"], float(x["weight"]), int(x["direction"]))
               for x in ens["selected_factors"]]
    panel = pd.read_parquet(ROOT / f"WL-data-final/panels/WL{wl}_full.parquet")
    panel["date"] = panel["date"].astype(str)
    px = panel[panel["asset_id"].isin(ASSETS)].pivot_table(
        index="date", columns="asset_id", values="close").sort_index()
    px = px.reindex(columns=ASSETS)
    return {"account": ac, "factors": factors, "px": px, "sandbox": sandbox}


def replay(wl: int, mode: str) -> dict:
    wl_data = load_worldline(wl)
    ac, factors, px = wl_data["account"], wl_data["factors"], wl_data["px"]
    r0 = ac["rebalance_history"][0]
    anchor_date = r0["date"]
    nav = float(r0["post_trade_nav"])
    w = {a: float(r0["target_weights"][a]) for a in ASSETS}

    decisions = [d for d in ac["decision_history"]
                 if d["date"] > anchor_date
                 and d.get("skip_reason") == "missing_forecast_proposal"]
    decisions.sort(key=lambda d: d["date"])

    daily_dates = list(px.index[px.index >= anchor_date])
    px0 = px.loc[anchor_date]
    shares = {a: nav * w[a] / px0[a] for a in ASSETS}

    records = []
    prev_date = anchor_date
    for dec in decisions:
        d = dec["date"]
        if d not in px.index:
            d = px.index[px.index <= d][-1]
        # drift to decision date
        for dt in [x for x in daily_dates if prev_date < x <= d]:
            row = px.loc[dt]
            nav = sum(shares[a] * row[a] for a in ASSETS)
        prev_date = d
        cur_w = {}
        for a in ASSETS:
            cur_w[a] = shares[a] * px.loc[d, a] / nav
        prop = dec["proposed_target_weights"]
        p = {a: float(prop[a]) for a in ASSETS}
        turnover = 0.5 * sum(abs(p[a] - cur_w[a]) for a in ASSETS)
        forecast, used = forecasts_at(px, factors, px.index[px.index < d][-1])
        edge_bps = 10000.0 * sum((p[a] - cur_w[a]) * forecast[a] for a in ASSETS)
        threshold_bps = COST_BPS * turnover
        executed = (mode == "all") or (edge_bps > threshold_bps + 1e-12)
        cost = 0.0
        if executed:
            cost = nav * turnover * COST_BPS / 10000.0
            nav = nav - cost
            shares = {a: nav * p[a] / px.loc[d, a] for a in ASSETS}
            w = p
        records.append({
            "date": d,
            "nav": round(nav, 2),
            "turnover": round(turnover, 6),
            "edge_bps": round(edge_bps, 4),
            "threshold_bps": round(threshold_bps, 4),
            "executed": executed,
            "cost": round(cost, 4),
            "used_factors": used,
        })

    # drift to end of panel (or last available)
    for dt in [x for x in daily_dates if prev_date < x]:
        row = px.loc[dt]
        nav = sum(shares[a] * row[a] for a in ASSETS)
    return {
        "wl": wl,
        "mode": mode,
        "anchor_date": anchor_date,
        "anchor_nav": float(r0["post_trade_nav"]),
        "final_date": daily_dates[-1],
        "final_nav": round(nav, 2),
        "executed_count": sum(1 for r in records if r["executed"]),
        "total_cost": round(sum(r["cost"] for r in records), 2),
        "n_decisions": len(records),
        "records": records,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--mode", choices=["gate", "all"], default="gate")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = {}
    for wl in args.wl:
        res = replay(wl, args.mode)
        out[wl] = res
        print(f"\n=== WL{wl} [{res['mode']}] ===")
        print(f"anchor {res['anchor_date']} NAV {res['anchor_nav']:,.2f}")
        print(f"decisions {res['n_decisions']}, executed {res['executed_count']}, "
              f"total cost {res['total_cost']:,.2f}")
        print(f"final {res['final_date']} NAV {res['final_nav']:,.2f}")
        print("recent:")
        for r in res["records"][-8:]:
            print(" ", r["date"], "nav", f"{r['nav']:,.2f}", "to", r["turnover"],
                  "edge", r["edge_bps"], "thr", r["threshold_bps"], "exe", r["executed"])
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
