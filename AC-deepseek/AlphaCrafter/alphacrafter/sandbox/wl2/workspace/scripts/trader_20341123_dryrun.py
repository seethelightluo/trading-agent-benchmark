"""Trader dry-run at block start 2034-11-23 (visible 11-22): compute target,
one-way turnover, gross edge and gate decision WITHOUT mutating state."""
import json, math, sys
sys.path.insert(0, ".")
from strategy import build_target, _load_ensemble, _current_weights, _regime, _closes

date_state = json.load(open("../persistent/date.json"))
print("current_date:", date_state["current_date"], "| visible_through:", date_state["visible_through"])

account = json.load(open("../persistent/account.json"))
assets = list(account.get("watch_list", []))
print("n_assets:", len(assets))

ensemble = _load_ensemble()
print("ensemble factors:", [(f["factor_id"], round(f["weight"], 4), f["direction"]) for f in ensemble])

cur_w = _current_weights(account, assets)
built = build_target(assets, date_state, ensemble, current_weights=cur_w)
if built is None:
    print("NO TARGET BUILT")
    sys.exit(0)
weights, forecast, used, meta = built

print("meta: risk=%.3f vix=%.1f m20=%.4f disp=%.4f n_factors=%d" % (
    meta["risk"], meta["vix"], meta["m20"], meta["disp"], meta["n_factors"]))
print("r20:", {a: round(v * 100, 1) for a, v in sorted(meta["r20"].items(), key=lambda kv: kv[1])})
print("cap_map:", meta["cap_map"])

print("\ntarget weights vs current:")
tot_t = 0.0
for a in assets:
    print("  %-10s target=%7.4f  cur=%7.4f  delta=%+7.4f  fcst=%+6.3f" % (
        a, weights[a], cur_w[a], weights[a] - cur_w[a], forecast[a]))
    tot_t += weights[a]
print("sum(target)=%.6f" % tot_t)

turn = sum(abs(weights[a] - cur_w[a]) for a in assets)
edge = sum((weights[a] - cur_w[a]) * forecast[a] for a in assets)
thresh = 0.0003 * turn
print("\none-way turnover: %.4f (%.2f%%)" % (turn, turn * 100))
print("gross edge: %.6f (%.2f bps)" % (edge, edge * 10000))
print("threshold 3bp*turnover: %.6f (%.2f bps)" % (thresh, thresh * 10000))
print("GATE: %s" % ("EXECUTE" if edge > thresh else "SKIP (edge <= 3bp*turnover)"))

# defensive vs aggressive breakdown
DEF = {"XAU", "US10Y", "CN10Y"}
AGG = {"SOX", "NDX", "ETH", "BTC", "000688.SH", "N225"}
print("\ndefensive wt: %.4f (cur %.4f) | aggressive wt: %.4f (cur %.4f)" % (
    sum(weights[a] for a in DEF), sum(cur_w[a] for a in DEF),
    sum(weights[a] for a in AGG), sum(cur_w[a] for a in AGG)))
