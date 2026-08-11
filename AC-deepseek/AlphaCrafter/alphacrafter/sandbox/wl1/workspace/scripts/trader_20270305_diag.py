"""Trader diagnostic 2027-03-05: verify 6-factor ensemble coverage & target weights."""
import json
import sys

sys.path.insert(0, ".")
import strategy as st

# Sanity: ensemble loaded
print("FACTORS from factor_ensemble.json:")
for fid, w, d in st.FACTORS:
    print(f"  {fid} w={w} dir={d}")
print("num factors:", len(st.FACTORS), "(cap 10 ok)" if len(st.FACTORS) <= 10 else "EXCEEDS CAP")

cur, tds = st._today_and_calendar()
print("current_date:", cur, "| should_propose:", st._should_propose(cur, tds))

# Fetch watchlist via account
from alphacrafter.sim.utils import get_account_dict
acc = get_account_dict()
assets = list(acc.get("watch_list", []))
print("watch_list:", assets)
print("num assets:", len(assets))

frames = st._fetch(assets)
missing = [a for a in assets if frames.get(a) is None]
print("missing frames:", missing)

scores, used = st._scores(frames, assets, cur)
print("factors used in score:", used, "(need >=5)")

# Per-factor coverage
for fid, w, d in st.FACTORS:
    vals = st._factor_values(frames, fid, cur)
    nv = sum(1 for v in vals.values() if v is not None)
    print(f"  {fid}: coverage {nv}/15")

regime = st._regime(frames, assets)
print("regime:", regime)

w = st._weights(scores, assets, regime)
f = st._forecasts(scores, assets)
tot = sum(w.values())
print("sum(w):", tot)
print("min w:", min(w.values()), "max w:", max(w.values()), "any neg:", any(v < 0 for v in w.values()))
order = sorted(assets, key=lambda a: (scores[a], a))
print("score ranking (top->bottom):")
for i, a in enumerate(order):
    print(f"  {i+1:2d}. {a:8s} score={scores[a]:.4f} w={w[a]:.4f} f={f[a]:+.4f}")

# defensive share
def_share = sum(w[a] for a in st.DEFENSIVE if a in assets)
print("defensive share (XAU/US10Y/CN10Y):", round(def_share, 4))
