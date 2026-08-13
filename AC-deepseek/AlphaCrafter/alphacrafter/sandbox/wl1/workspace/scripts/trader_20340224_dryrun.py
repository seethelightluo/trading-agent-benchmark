"""Trader dry-run verification for 2034-02-24 (no execution).

Verifies: ensemble loaded from factor_ensemble.json matches strategy,
proposal gating, regime assessment, score computation and final target
weights WITHOUT calling rebalance_to_weights / add_order / step.
"""
import json
import numpy as np
import strategy as st

# --- 1. ensemble loaded by strategy == ensemble file ---
ens_file = json.load(open("factor_ensemble.json"))["selected_factors"]
ens_file_ids = [x["factor_id"] for x in ens_file]
print("== ensemble (file) ==")
for x in ens_file:
    print(f"  {x['factor_id']:40s} w={x['weight']:.2f} dir={x['direction']:+d}")
print("== ensemble (strategy.FACTORS at import) ==")
for fid, w, d in st.FACTORS:
    print(f"  {fid:40s} w={w:.2f} dir={d:+d}")
assert [f for f, _, _ in st.FACTORS] == ens_file_ids, "FACTORS mismatch!"

# --- 2. proposal gating ---
cur, tds = st._today_and_calendar()
print("\ncurrent_date:", cur)
print("should_propose:", st._should_propose(cur, tds))
print("is_rebalance_day (grid):", st._is_rebalance_day(cur, tds))
print("last_proposal_date:", st._last_proposal_date(tds))

# --- 3. data fetch + scores ---
account = __import__("alphacrafter.sim.utils", fromlist=["get_account_dict"]).get_account_dict()
assets = list(account.get("watch_list", []))
print("\nassets n =", len(assets))
frames = st._fetch(assets)
missing = [a for a in assets if frames.get(a) is None]
print("missing/degraded frames:", missing)
scores, used = st._scores(frames, assets, cur)
print("factors used in score:", used)
scores = st._de_rank_value_traps(scores, frames, assets, cur)
regime = st._regime(frames, assets)
print("regime:", regime)

# --- 4. weight pipeline ---
w = st._weights(scores, assets, regime)
w = st._composite_top2_cap(w, assets, scores)
w = st._composite_ma_guard(w, frames, assets)
w = st._ma_guard(w, frames, assets, cur)
for _ in range(6):
    w = st._commod_cap(w, assets)
    w = st._crypto_cap(w, assets)
    w = st._china_cap(w, assets)
    w = st._composite_top2_cap(w, assets, scores)

print("\n== target weights (dry-run) ==")
tot = 0.0
for a in sorted(assets, key=lambda x: -w[x]):
    print(f"  {a:10s} {w[a]*100:6.2f}%")
    tot += w[a]
print("sum:", round(tot, 8), "min:", min(w.values()), "max:", max(w.values()))
assert abs(tot - 1.0) < 1e-6, "weights do not sum to 1!"

# pair caps
crypto = sum(w[a] for a in st.CRYPTO if a in w)
comm = sum(w[a] for a in st.CYCLICAL_COMMOD if a in w)
china = sum(w[a] for a in st.CHINA_EQ if a in w)
print(f"\ncrypto pair {crypto*100:.2f}% (cap 12) | commod pair {comm*100:.2f}% (cap 12) | china pair {china*100:.2f}% (cap 12)")

# top-2 composite check
order = sorted(assets, key=lambda a: (scores[a], a))
top2 = order[-2:]
print("top-2 composite names:", top2, "weights:", [round(w[a]*100, 2) for a in top2])

# --- 5. forecasts ---
f = st._forecasts(scores, assets)
print("\nforecast range:", round(min(f.values()), 4), "..", round(max(f.values()), 4))
print("\nDRY-RUN OK - no execution performed")
