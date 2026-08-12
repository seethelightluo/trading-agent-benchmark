#!/usr/bin/env python3
"""Offline recompute + account wiring for Terra wl2/wl3 (no API calls).

Replay contract (documented in runAC.md, 2026-08-12):
- Anchor: the only verified executed state, account.rebalance_history[0]
  (2026-07-30 initial allocation, pre/post NAV recorded at open).
- Decisions: recorded decision_history.proposed_target_weights from the old
  era (skip_reason == "missing_forecast_proposal").  The trader LLM's
  proposal is authoritative: each proposal executes with a 3bps cost.
- Execution price: OPEN of the decision date (mirrors rebalance_to_weights).
- NAV is marked at CLOSE each day (mirrors the sim's post_tick accounting).
- After the last old-era decision the portfolio drifts (no rebalance) to the
  current sim date from date.json, where the account is re-wired for resume.

Usage:
  ac_recompute_apply.py --wl 2 3 --mode all          # dry-run report
  ac_recompute_apply.py --wl 2 3 --mode all --apply  # backup + write accounts
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lxx/trade-agent-benchmark")
ALPHA = ROOT / "agent-framework/AlphaCrafter/alphacrafter"
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
          "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
COST_BPS = 3.0
PORTFOLIO_CONTRACT_VERSION = "ac-worldline-v2-migration-gate"
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


def forecasts_at(px: pd.DataFrame, ensemble: list, visible_through: str) -> tuple[dict, int]:
    scores = {a: 0.0 for a in ASSETS}
    used = 0
    for fid, w, direction in ensemble:
        vals: dict[str, float | None] = {}
        ok = 0
        for a in ASSETS:
            try:
                hist = px.loc[:visible_through, ("close", a)].dropna().tail(DATA_DAYS)
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
    dstate = json.load(open(sandbox / "persistent/date.json"))
    ens = json.load(open(sandbox / "workspace/factor_ensemble.json"))
    factors = [(x["factor_id"], float(x["weight"]), int(x["direction"]))
               for x in ens["selected_factors"]]
    panel = pd.read_parquet(ROOT / f"WL-data-final/panels/WL{wl}_full.parquet")
    panel["date"] = panel["date"].astype(str)
    px = panel[panel["asset_id"].isin(ASSETS)].pivot_table(
        index="date", columns="asset_id", values=["open", "close"]).sort_index()
    return {"account": ac, "date": dstate, "factors": factors,
            "px": px, "sandbox": sandbox}


def _price(px: pd.DataFrame, field: str, date: str, asset: str) -> float:
    try:
        v = float(px.loc[date, (field, asset)])
        if math.isfinite(v) and v > 0:
            return v
    except (KeyError, TypeError):
        pass
    prior = [d for d in px.index if d < date and math.isfinite(float(px.loc[d, (field, asset)]))]
    if not prior:
        raise ValueError(f"no {field} price for {asset} on/before {date}")
    return float(px.loc[prior[-1], (field, asset)])


def recompute(wl: int, mode: str) -> dict:
    wl_data = load_worldline(wl)
    ac, dstate, factors, px = (wl_data["account"], wl_data["date"],
                               wl_data["factors"], wl_data["px"])
    r0 = ac["rebalance_history"][0]
    anchor_date = r0["date"]
    anchor_pre = float(r0["pre_trade_nav"])
    anchor_post = float(r0["post_trade_nav"])
    t0 = {a: float(r0["target_weights"][a]) for a in ASSETS}

    decisions = [d for d in ac["decision_history"]
                 if d["date"] > anchor_date
                 and d.get("skip_reason") == "missing_forecast_proposal"]
    decisions.sort(key=lambda d: d["date"])

    # shares right after the anchor rebalance (executed at open of anchor_date)
    shares = {a: anchor_post * t0[a] / _price(px, "open", anchor_date, a) for a in ASSETS}
    anchor_cost = float(r0.get("cost", 0.0))

    rebalances = [{
        "date": anchor_date,
        "initial_allocation": True,
        "pre_trade_nav": anchor_pre,
        "post_trade_nav": anchor_post,
        "transferred_notional": float(r0.get("transferred_notional", 0.0)),
        "cost_bps": float(r0.get("cost_bps", COST_BPS)),
        "cost": anchor_cost,
        "target_weights": t0,
    }]
    records: list[dict] = []
    prev_exec_date = anchor_date
    total_cost = anchor_cost

    for dec in decisions:
        d = dec["date"]
        nav_open = sum(shares[a] * _price(px, "open", d, a) for a in ASSETS)
        cur_w = {a: shares[a] * _price(px, "open", d, a) / nav_open for a in ASSETS}
        prop = {a: float(dec["proposed_target_weights"][a]) for a in ASSETS}
        turnover = 0.5 * sum(abs(prop[a] - cur_w[a]) for a in ASSETS)
        visible = [x for x in px.index if x < d]
        forecast, used = forecasts_at(px, factors, visible[-1]) if visible else ({}, 0)
        edge_bps = 10000.0 * sum((prop[a] - cur_w[a]) * forecast[a] for a in ASSETS)
        threshold_bps = COST_BPS * turnover
        execute = (mode == "all") or (edge_bps > threshold_bps + 1e-12)
        cost = 0.0
        if execute:
            cost = nav_open * turnover * COST_BPS / 10000.0
            total_cost += cost
            shares = {a: (nav_open - cost) * prop[a] / _price(px, "open", d, a) for a in ASSETS}
            prev_exec_date = d
            rebalances.append({
                "date": d,
                "initial_allocation": False,
                "pre_trade_nav": nav_open,
                "post_trade_nav": nav_open - cost,
                "transferred_notional": nav_open * turnover,
                "cost_bps": COST_BPS,
                "cost": cost,
                "target_weights": prop,
            })
        nav_close = sum(shares[a] * _price(px, "close", d, a) for a in ASSETS)
        records.append({
            "date": d,
            "executed": execute,
            "nav_open": nav_open,
            "nav_close": nav_close,
            "turnover": turnover,
            "edge_bps": edge_bps,
            "threshold_bps": threshold_bps,
            "cost": cost,
            "used_factors": used,
            "current_weights": cur_w,
            "proposed": prop,
            "forecast": forecast,
        })

    # drift to the current sim date
    cur_date = str(dstate["current_date"])
    dates = [x for x in px.index if anchor_date <= x <= cur_date]
    for dt in dates:
        nav = sum(shares[a] * _price(px, "close", dt, a) for a in ASSETS)
    final_nav = nav
    final_w = {a: shares[a] * _price(px, "close", cur_date, a) / final_nav for a in ASSETS}

    return {
        "wl": wl, "mode": mode,
        "anchor_date": anchor_date, "anchor_post_nav": anchor_post,
        "last_decision_date": records[-1]["date"],
        "current_date": cur_date,
        "final_nav": final_nav,
        "final_weights": final_w,
        "shares": shares,
        "executed_count": sum(1 for r in records if r["executed"]),
        "n_decisions": len(records),
        "total_cost": total_cost,
        "last_exec_date": prev_exec_date,
        "rebalances": rebalances,
        "records": records,
        "initial_capital": float(ac.get("initial_capital", 1_000_000.0)),
        "watch_list": list(ac.get("watch_list", ASSETS)),
    }


def build_account(wl_data: dict, res: dict) -> dict:
    px = wl_data["px"]
    cur = res["current_date"]
    positions = []
    for a in ASSETS:
        qty = res["shares"][a]
        if qty <= 1e-12:
            continue
        exec_date = res["last_exec_date"]
        cost_px = _price(px, "open", exec_date, a)
        cur_px = _price(px, "close", cur, a)
        mv = qty * cur_px
        positions.append({
            "symbol": a, "direction": "LONG", "quantity": qty,
            "available_quantity": qty, "cost_price": cost_px,
            "current_price": cur_px, "market_value": mv,
            "profit_loss": mv - qty * cost_px,
            "profit_loss_rate": mv / max(qty * cost_px, 1e-12) - 1.0,
        })
    nav = res["final_nav"]
    initial = res["initial_capital"]
    ac = {
        "initial_capital": initial,
        "total_assets": nav,
        "net_assets": nav,
        "available_cash": 0.0,
        "market_value": nav,
        "total_profit_loss": nav - initial,
        "total_profit_loss_rate": nav / initial - 1.0,
        "gross_position_rate": 1.0,
        "net_position_rate": 1.0,
        "positions": positions,
        "orders": [],
        "watch_list": res["watch_list"],
        "portfolio_initialized": True,
        "portfolio_contract_version": PORTFOLIO_CONTRACT_VERSION,
        "last_rebalance_date": res["last_exec_date"],
        "last_target_weights": res["rebalances"][-1]["target_weights"],
        "last_proposed_target_weights": res["records"][-1]["proposed"],
        "last_executed_target_weights": res["rebalances"][-1]["target_weights"],
        "cumulative_transaction_cost": res["total_cost"],
        "rebalance_history": [],
        "decision_history": [],
    }
    for rb in res["rebalances"]:
        ac["rebalance_history"].append({
            **{k: v for k, v in rb.items()},
            "proposed_target_weights": rb["target_weights"],
            "executed_target_weights": rb["target_weights"],
            "forecast_returns": {},
            "factor_ids": [],
            "horizon_days": 10,
            "one_way_turnover": 0.5 * sum(
                abs(rb["target_weights"][a] - res["records"][0]["current_weights"].get(a, 0.0))
                for a in ASSETS) if len(res["rebalances"]) == 1 else 0.0,
            "gross_edge_bps": 0.0,
            "decision_edge_threshold_bps": 0.0,
            "actual_cost": rb["cost"],
            "portfolio_contract_version": PORTFOLIO_CONTRACT_VERSION,
            "executed": True,
            "skip_reason": "",
        })
    # one_way_turnover for non-anchor rebalances from the replay records
    by_date = {r["date"]: r for r in res["records"]}
    for i, rb in enumerate(ac["rebalance_history"]):
        rec = by_date.get(rb["date"])
        if rec is not None:
            rb["one_way_turnover"] = rec["turnover"]
            rb["gross_edge_bps"] = rec["edge_bps"]
            rb["decision_edge_threshold_bps"] = rec["threshold_bps"]
            rb["forecast_returns"] = rec["forecast"]
        if i == 0:
            rb["initial_allocation"] = True
            rb["one_way_turnover"] = float(res["rebalances"][0].get("transferred_notional", 0.0)) / max(rb["pre_trade_nav"], 1e-9)
    for rec in res["records"]:
        ac["decision_history"].append({
            "date": rec["date"],
            "initial_allocation": False,
            "current_weights": rec["current_weights"],
            "proposed_target_weights": rec["proposed"],
            "executed_target_weights": rec["proposed"] if rec["executed"] else rec["current_weights"],
            "forecast_returns": rec["forecast"],
            "factor_ids": [],
            "horizon_days": 10,
            "one_way_turnover": rec["turnover"],
            "gross_edge_bps": rec["edge_bps"],
            "decision_edge_threshold_bps": rec["threshold_bps"],
            "actual_cost": rec["cost"],
            "portfolio_contract_version": PORTFOLIO_CONTRACT_VERSION,
            "executed": rec["executed"],
            "skip_reason": "" if rec["executed"] else "gross_edge_not_above_migration_cost",
        })
    return ac


def verify_account(wl: int, ac: dict, res: dict) -> list[str]:
    checks = []
    px = load_worldline(wl)["px"]
    cur = res["current_date"]
    nav_pos = sum(float(p["quantity"]) * float(p["current_price"]) for p in ac["positions"])
    checks.append(f"positions NAV {nav_pos:,.2f} vs net_assets {ac['net_assets']:,.2f} diff {abs(nav_pos-ac['net_assets']):.4f}")
    wsum = sum(ac["last_executed_target_weights"][a] for a in ASSETS)
    checks.append(f"last executed weights sum {wsum:.8f}")
    checks.append(f"contract {ac['portfolio_contract_version']} initialized={ac['portfolio_initialized']}")
    checks.append(f"rebalance_history {len(ac['rebalance_history'])}, decision_history {len(ac['decision_history'])}, cost {ac['cumulative_transaction_cost']:,.2f}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--mode", choices=["all", "gate"], default="all")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    for wl in args.wl:
        res = recompute(wl, args.mode)
        print(f"\n=== WL{wl} [{res['mode']}] ===")
        print(f"anchor {res['anchor_date']} post-NAV {res['anchor_post_nav']:,.2f}")
        print(f"decisions {res['n_decisions']}, executed {res['executed_count']}, "
              f"total cost {res['total_cost']:,.2f}")
        print(f"last exec {res['last_exec_date']}, current {res['current_date']}, "
              f"final NAV {res['final_nav']:,.2f}")
        for r in res["records"][-5:]:
            print(f"  {r['date']} exe={r['executed']} to={r['turnover']:.4f} "
                  f"edge={r['edge_bps']:.2f} thr={r['threshold_bps']:.3f} "
                  f"cost={r['cost']:.2f} nav_close={r['nav_close']:,.2f}")
        if args.apply:
            wl_data = load_worldline(wl)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = ROOT / "backups" / f"20260812_pre_recompute_{stamp}" / f"wl{wl}"
            src = wl_data["sandbox"]
            backup_dir.mkdir(parents=True, exist_ok=True)
            for part in ("persistent", "workspace", "logs", "config"):
                p = src / part
                if p.exists():
                    shutil.copytree(p, backup_dir / part)
            ac = build_account(wl_data, res)
            checks = verify_account(wl, ac, res)
            for c in checks:
                print("  verify:", c)
            target = src / "persistent" / "account.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(ac, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(target)
            (src / "workspace" / "trader_state.json").write_text(
                json.dumps({"last_proposal_date": res["last_decision_date"]}, ensure_ascii=False),
                encoding="utf-8")
            print(f"  wrote {target} (backup: {backup_dir})")


if __name__ == "__main__":
    main()
