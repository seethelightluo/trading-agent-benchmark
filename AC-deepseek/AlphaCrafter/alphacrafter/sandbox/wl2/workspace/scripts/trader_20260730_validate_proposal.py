"""Trader validation: compute 2026-07-30 proposal offline (no execution)."""
import json
from pathlib import Path
import strategy as S

date_state = json.loads(Path("../persistent/date.json").read_text())
acc = json.loads(Path("../persistent/account.json").read_text())
assets = acc["watch_list"]
ens = S._load_ensemble()
built = S.build_target(assets, date_state, ens)
assert built is not None, "build failed"
weights, forecast, used, meta = built

print("factors used:", used)
print("meta:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta.items()})
print("\ncurrent vs proposed weights:")
cur = {a: 0.0 for a in assets}
for p in acc["positions"]:
    cur[p["symbol"]] = p["market_value"] / acc["total_assets"]
tw = 0.0
for a in assets:
    tw += weights[a]
    print(f"{a:10s} cur={cur[a]*100:6.2f}%  prop={weights[a]*100:6.2f}%  d={ (weights[a]-cur[a])*100:+6.2f}%  z={meta['z'][a]:+.2f}  f={forecast[a]*100:+.2f}%")
print("\nsum weights:", round(tw, 6), "| cap ok:", all(w <= 0.17 + 1e-9 for w in weights.values()),
      "| floor ok:", all(w >= 0.012 - 1e-9 for w in weights.values()))

# one-way turnover and gross edge (bps) vs current
turn = 0.5 * sum(abs(weights[a] - cur[a]) for a in assets)
edge = sum((weights[a] - cur[a]) * forecast[a] for a in assets)
nav = acc["total_assets"]
print(f"\none-way turnover: {turn*100:.2f}%  (migrated notional ~ {turn*nav:,.0f})")
print(f"gross edge: {edge*10000:.1f} bps of NAV | threshold: {turn*3:.1f} bps | execute: {edge*10000 > turn*3}")
