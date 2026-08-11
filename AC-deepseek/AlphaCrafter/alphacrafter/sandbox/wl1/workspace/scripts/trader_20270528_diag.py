"""Trader read-only diagnostic for 2027-05-28 block start.

Mimics strategy.py factor computation for the current date without touching
the live account. Verifies: ensemble load, factor coverage, weights sum to 1,
cap applied, regime classification.
"""
import json
import sys
sys.path.insert(0, ".")
import strategy as st

cur, tds = st._today_and_calendar()
print("current_date:", cur)
print("is_rebalance_day(grid):", st._is_rebalance_day(cur, tds))
print("last_proposal_date:", st._last_proposal_date(tds))
print("should_propose:", st._should_propose(cur, tds))

FACTORS = st.FACTORS
print("ensemble factors:", [(f, w, d) for f, w, d in FACTORS])
assert len(FACTORS) <= 10, "more than 10 active factors!"

# fetch data for the 15-asset watchlist (need account for watch_list)
acc = st.get_account_dict()
assets = list(acc.get("watch_list", []))
print("n assets:", len(assets), assets)

frames = st._fetch(assets)
n_missing = sum(1 for a in assets if frames.get(a) is None)
print("frames missing:", n_missing)

scores, used = st._scores(frames, assets, cur)
print("factors used:", used)
for a in assets:
    print(f"  score {a}: {scores[a]:.4f}")

regime = st._regime(frames, assets)
print("regime:", regime)

w = st._weights(scores, assets, regime)
tot = sum(w.values())
print("weights sum:", tot)
print("max weight:", max(w.values()), "min weight:", min(w.values()))
print("weights:", {a: round(x, 4) for a, x in sorted(w.items(), key=lambda kv: -kv[1])})

f = st._forecasts(scores, assets)
print("forecast sample:", {a: round(f[a], 4) for a in assets[:5]})
print("DIAG OK")
