"""Trader validation: recompute strategy target weights as of 2026-07-16 (data through 07-15)."""
import json
import sys
sys.path.insert(0, ".")
import strategy as st

# Recompute exactly as the hook would
cur, tds = st._today_and_calendar()
print("current:", cur, "| is_rebalance_day:", st._is_rebalance_day(cur, tds))

acc = json.load(open("../persistent/account.json"))
assets = acc["watch_list"]
frames = st._fetch(assets)
scores, used = st._scores(frames, assets)
print("used factors:", used)
regime = st._regime(frames, assets)
print("regime:", regime)

w = st._weights(scores, assets, regime)
f = st._forecasts(scores, assets)

print("\n--- recomputed target weights ---")
for a in assets:
    print(f"{a:10s} w={w[a]:.6f} score={scores[a]:.4f} fc={f[a]:+.4f}")

tot = sum(w.values())
print("\nsum(w) =", round(tot, 10), "| min:", min(w.values()), "| max:", max(w.values()))
assert abs(tot - 1.0) < 1e-9 and all(v >= 0 for v in w.values())

ltw = acc["last_target_weights"]
diff = {a: w[a] - ltw[a] for a in assets}
maxdiff = max(abs(v) for v in diff.values())
print("max |w - last_target_weights|:", maxdiff)

# turnover vs current holdings (by market value)
mv = {p["symbol"]: p["market_value"] for p in acc["positions"]}
nav = acc["total_assets"]
print("\n--- turnover vs current holdings ---")
turnover = 0.0
for a in assets:
    cur_w = mv.get(a, 0.0) / nav
    turnover += abs(w[a] - cur_w)
print("one-way turnover (vs current):", round(turnover / 2.0, 6))

# gross edge estimate: forecast-return weighted by weight change
edge = sum(f[a] * (w[a] - mv.get(a, 0.0) / nav) for a in assets)
print("gross edge (sum fc*dw):", round(edge, 6), "| gate threshold (turnover*3bp):", round((turnover / 2.0) * 0.0003, 6))
