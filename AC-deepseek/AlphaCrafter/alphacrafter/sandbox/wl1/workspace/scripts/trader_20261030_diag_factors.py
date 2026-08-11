"""Trader diagnostic: verify new 6-factor ensemble computes on current date."""
import json, sys
sys.path.insert(0, ".")
import strategy as st

cur, tds = st._today_and_calendar()
print("current_date:", cur)
print("is rebalance day:", st._is_rebalance_day(cur, tds))
print("factors:", [f[0] for f in st.FACTORS])

acc = __import__("alphacrafter.sim.utils", fromlist=["get_account_dict"]).get_account_dict()
assets = list(acc.get("watch_list", []))
print("n assets:", len(assets))

frames = st._fetch(assets)
missing = [a for a, df in frames.items() if df is None]
print("missing frames:", missing)

scores, used = st._scores(frames, assets, cur)
print("used factors:", used, "of", len(st.FACTORS))
for fid, w, d in st.FACTORS:
    vals = st._factor_values(frames, fid, cur)
    nv = sum(1 for v in vals.values() if v is not None)
    print(f"  {fid:32s} w={w:.2f} d={d:+d} valid={nv}/15")

print("\nranked scores (high->low):")
for a in sorted(assets, key=lambda x: -scores[x]):
    print(f"  {a:10s} {scores[a]:.4f}")

regime = st._regime(frames, assets)
w = st._weights(scores, assets, regime)
print("\nregime:", regime)
print("sum w:", round(sum(w.values()), 10))
print("min w:", round(min(w.values()), 6), "max w:", round(max(w.values()), 6))
print("target:", {a: round(w[a], 4) for a in sorted(w, key=lambda x: -w[x])})
