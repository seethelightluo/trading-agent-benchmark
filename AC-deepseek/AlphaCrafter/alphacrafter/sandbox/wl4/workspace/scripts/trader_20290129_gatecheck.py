"""Reconstruct the v24 target proposal as of 2029-01-15 (block start) and
compare with actual holdings to determine if the gate executed the migration.
"""
import json
import sys
sys.path.insert(0, ".")
import strategy as st
from alphacrafter.sim.utils import get_account_dict

CUTOFF = "2029-01-15"

# Build the same inputs compute_target uses but truncated at CUTOFF close.
assets = list(get_account_dict()["watch_list"])
frames = {a: st.stock(a) for a in assets}
closes = {}
for a, f in frames.items():
    if f is not None and "close" in f:
        c = f["close"].astype(float)
        c.index = f["date"] if "date" in f else c.index
        c = c[c.index <= CUTOFF]
        closes[a] = c
    else:
        closes[a] = None

vol = {a: (f["volume"].astype(float) if f is not None and "volume" in f else None)
       for a, f in frames.items()}

stale = {a for a in assets if st.is_stale(closes, a)}
live = [a for a in assets if a not in stale]

usable = [c.pct_change().rename(a) for a, c in closes.items()
          if a in live and c is not None and len(c) >= 30]
panel = (st.pd.concat(usable, axis=1, join="inner").dropna().tail(130)
         if len(usable) >= 8 else st.pd.DataFrame())

ef = st.index("EURUSD")
eurusd_ret = (ef["close"].astype(float).pct_change() if ef is not None else None)
cn10y_ret = closes["CN10Y"].pct_change() if closes.get("CN10Y") is not None else None

ens = st.load_ensemble()
print("ensemble:", ens)
vals = st.compute_factor_values(assets, stale, closes, panel, eurusd_ret, cn10y_ret, vol)

score = {a: 0.0 for a in assets}
for fid, wgt, drc in ens:
    r = st.ranks(vals.get(fid, {}), assets)
    for a in assets:
        score[a] += wgt * drc * r[a]

s_vals = [score[a] for a in assets]
lo, hi = min(s_vals), max(s_vals)
vol20 = {a: max(float(panel[a].tail(20).std()), 0.004) for a in live}
med_vol = sorted(vol20.values())[len(vol20) // 2] if vol20 else 0.01
for a in stale:
    vol20[a] = med_vol

base = {a: st.FLOOR + st.SPREAD * ((score[a] - lo) / (hi - lo + 1e-12)) for a in assets}
for a, m in st.DEFENSIVE_MULT.items():
    if a in base:
        base[a] *= m
tilted = {a: base[a] / (vol20[a] ** st.VOL_EXP) for a in assets}
w = st.capped_normalize(tilted)

# Actual holdings weights
pos = {p["symbol"]: p for p in get_account_dict()["positions"]}
tot_mv = sum(p["market_value"] for p in pos.values())
actual = {s: p["market_value"] / tot_mv for s, p in pos.items()}

print("\n== PROPOSAL (01-15) vs ACTUAL ==")
for a in assets:
    print(f"{a:10s} prop={w.get(a,0):.4f} actual={actual.get(a,0):.4f} diff={w.get(a,0)-actual.get(a,0):+.4f}")
print("prop sum:", round(sum(w.values()), 6), "actual sum:", round(sum(actual.values()), 6))

# One-way turnover estimate
oneway = sum(abs(w.get(a, 0) - actual.get(a, 0)) for a in assets) / 2.0
print("one-way turnover:", round(oneway, 5))
