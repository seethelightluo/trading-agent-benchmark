"""Trader validation for 2027-04-16 cycle: check strategy imports, ensemble
support, and compute the proposed target vector invariants."""
import json
import math
import sys
sys.path.insert(0, ".")
import strategy as st

print("== FACTORS loaded by strategy ==")
for fid, w, d in st.FACTORS:
    print(f"  {fid:32s} w={w:.3f} dir={d:+d}")
assert len(st.FACTORS) <= 10, "factor cap exceeded"
assert abs(sum(w for _, w, _ in st.FACTORS) - 1.0) < 1e-6, "weights != 1"

ens = json.load(open("factor_ensemble.json"))
ens_ids = {x["factor_id"] for x in ens["selected_factors"]}
strat_ids = {fid for fid, _, _ in st.FACTORS}
assert ens_ids == strat_ids, f"mismatch: {ens_ids ^ strat_ids}"
print("== ensemble == strategy: MATCH ==")

cur, tds = st._today_and_calendar()
print("current_date:", cur)
print("should_propose:", st._should_propose(cur, tds))
print("last_proposal:", st._last_proposal_date(tds))

from alphacrafter.sim.utils import get_account_dict
acc = get_account_dict()
assets = list(acc["watch_list"])
print("n assets:", len(assets))

frames = st._fetch(assets)
n_missing = sum(1 for a in assets if frames.get(a) is None)
print("missing frames:", n_missing)

# per-factor support check
for fid, w, d in st.FACTORS:
    vals = st._factor_values(frames, fid, cur)
    nv = sum(1 for v in vals.values() if v is not None)
    print(f"  factor {fid:32s} valid={nv:2d}/15")

scores, used = st._scores(frames, assets, cur)
print("factors used in score:", used, "/", len(st.FACTORS))

regime = st._regime(frames, assets)
print("regime:", regime)

w = st._weights(scores, assets, regime)
print("== proposed weights ==")
tot = 0.0
for a in sorted(w, key=lambda x: -w[x]):
    print(f"  {a:10s} {w[a]:.4f}")
    tot += w[a]
print("sum(w):", round(tot, 6))
assert abs(tot - 1.0) < 1e-6, "sum != 1"
assert all(v >= 0 for v in w.values()), "negative weight!"
assert all(math.isfinite(v) for v in w.values()), "non-finite weight!"
assert set(w.keys()) == set(assets), "asset set mismatch"
print("ALL INVARIANTS OK: 15 assets, non-negative, finite, sum=1")

f = st._forecasts(scores, assets)
print("forecast sample:", {a: round(f[a], 4) for a in list(f)[:5]})
print("VALIDATION PASSED")
