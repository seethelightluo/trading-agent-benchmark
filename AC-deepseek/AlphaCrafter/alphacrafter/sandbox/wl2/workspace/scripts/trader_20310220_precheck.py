"""Trader pre-check 2031-02-20: simulate the block-start decision the hook will
make at 02-20 (visible through 02-19) using the current strategy build_target.
Predict target, turnover vs current weights, gross edge, and gate verdict.
Read-only: does not mutate account/date.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import strategy as S

BASE = Path(__file__).resolve().parent.parent

with open(BASE.parent / "persistent" / "date.json") as f:
    date_state = json.load(f)
with open(BASE.parent / "persistent" / "account.json") as f:
    account = json.load(f)

print("current_date:", date_state["current_date"], "| visible_through:", date_state["visible_through"])

# block-start check
trading_days = date_state["trading_days"]
weekdays = [x for x in trading_days if __import__("datetime").date.fromisoformat(x).weekday() < 5]
cur = date_state["current_date"]
k = weekdays.index(cur) - weekdays.index(S.ONLINE_START)
print("k (trading days since online start):", k, "| k % BLOCK:", k % S.BLOCK, "-> block start:", k % S.BLOCK == 0)

assets = list(account.get("watch_list", []))
if len(assets) != 15:
    assets = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
              "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
print("assets:", len(assets))

ensemble = S._load_ensemble()
print("ensemble factors:", [(e["factor_id"], round(e["weight"], 4), e["direction"]) for e in ensemble])

cur_w = S._current_weights(account, assets)
built = S.build_target(assets, date_state, ensemble, current_weights=cur_w)
if built is None:
    print("build_target -> None")
    sys.exit(0)
weights, forecast, used, meta = built
tot = sum(weights.values())
print("target sum:", round(tot, 6), "| n_factors used:", len(used), used)
print("risk:", round(meta["risk"], 3), "vix:", round(meta["vix"], 2), "m20:", round(meta["m20"], 5), "disp:", round(meta["disp"], 4))
print("r20:", {a: round(v * 100, 1) for a, v in meta["r20"].items()})
print("cap_map:", meta["cap_map"])
print()
print(f"{'asset':10s} {'cur_w':>7s} {'target':>7s} {'delta':>7s} {'forecast':>9s} {'z':>6s}")
edge = 0.0
turn = 0.0
for a in assets:
    c = cur_w.get(a, 0.0)
    t = weights[a]
    d = t - c
    turn += abs(d)
    edge += forecast[a] * d
    print(f"{a:10s} {c*100:6.2f}% {t*100:6.2f}% {d*100:+6.2f}% {forecast[a]*100:+8.2f}% {meta['z'][a]:+6.2f}")
print()
print("one-way turnover: {:.2%}".format(turn))
print("gross edge (sum fc*d): {:.4%}".format(edge))
thresh = turn * 0.0003
print("gate threshold (turn*3bp): {:.4%}".format(thresh))
print("GATE VERDICT:", "EXECUTE" if edge > thresh else "SKIP (persist proposal)")
print("total transferred notional est:", round(turn * account["total_assets"]))
