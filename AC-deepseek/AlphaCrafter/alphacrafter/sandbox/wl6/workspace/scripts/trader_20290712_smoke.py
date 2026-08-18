"""Smoke test: verify strategy computes all 7 selected ensemble factors on real data."""
import sys
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_account_dict
import strategy as st

assets = list(get_account_dict()["watch_list"])
print("watchlist:", assets, "n =", len(assets))
frames = {a: st.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None) for a, f in frames.items()}
vf = st.index("VIX")
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
raw = st.compute_raw_factors(closes, vix_close, assets)

ens = st.load_ensemble()
print("ensemble factors:", [f["factor_id"] for f in ens])
missing = [f["factor_id"] for f in ens if f["factor_id"] not in raw]
print("MISSING:", missing if missing else "none")
for fid in [f["factor_id"] for f in ens]:
    vals = {a: v for a, v in raw[fid].items() if v is not None}
    print(f"  {fid}: {len(vals)}/15 valid")
