"""Trader validation: dry-run the strategy pipeline on 2026-07-16 data."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import numpy as np
import strategy as S
from alphacrafter.sim.utils import get_account_dict

account = get_account_dict()
assets = list(account.get("watch_list", []))
print("n_assets:", len(assets))
print("assets:", assets)

obs_only = {"DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"}
assert len(assets) == 15 and len(set(assets)) == 15
assert not (set(assets) & obs_only), "observation-only asset leaked into tradable set"

frames = S._fetch(assets)
missing = [a for a in assets if frames.get(a) is None]
print("missing frames:", missing)
assert not missing

scores, used = S._scores(frames, assets)
print("factors used:", used, "of", len(S.FACTORS))
for fid, w, d in S.FACTORS:
    vals = S._factor_values(frames, fid)
    n_valid = sum(1 for v in vals.values() if v is not None)
    print(f"  {fid:35s} w={w:.4f} dir={d:+d} valid={n_valid}/15")

regime = S._regime(frames, assets)
w = S._weights(scores, assets, regime)
f = S._forecasts(scores, assets)

tot = sum(w.values())
print("regime:", regime)
print("sum(w) =", tot)
print("min(w) =", min(w.values()), " max(w) =", max(w.values()))
assert abs(tot - 1.0) < 1e-6
assert all(math.isfinite(v) and v >= 0 for v in w.values())
assert set(w) == set(assets)

ranked = sorted(assets, key=lambda a: w[a], reverse=True)
print("top weights:", [(a, round(w[a], 4)) for a in ranked[:6]])
print("bottom weights:", [(a, round(w[a], 4)) for a in ranked[-4:]])
print("defensive (XAU/US10Y/CN10Y):",
      [(a, round(w[a], 4)) for a in ["XAU", "US10Y", "CN10Y"]])
print("forecast range:", round(min(f.values()), 4), round(max(f.values()), 4))

mean_f = float(np.mean([f[a] for a in assets]))
edge = sum(w[a] * f[a] for a in assets) - mean_f
print("forecast-weighted gross edge:", round(edge, 6))
