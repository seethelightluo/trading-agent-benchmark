"""Trader sanity check: verify new factor raw values compute (non-neutral)."""
from math import isfinite
from alphacrafter.sim.utils import get_account_dict
import strategy as st

assets = list(get_account_dict()["watch_list"])
frames = {a: st.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
          for a, f in frames.items()}
vf = st.index("VIX")
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
print("assets:", len(assets), "usable closes:", sum(1 for c in closes.values() if c is not None))
print("vix_close last:", None if vix_close is None else float(vix_close.iloc[-1]))
raw = st.compute_raw_factors(closes, vix_close, assets)
for fid, vals in raw.items():
    valid = {a: v for a, v in vals.items() if v is not None and isfinite(v)}
    print(f"{fid}: valid={len(valid)}/15  sample={ {a: round(v,4) for a, v in list(valid.items())[:5]} }")
# neutral fraction
for fid, vals in raw.items():
    neutral = sum(1 for v in vals.values() if v is None or not isfinite(v))
    print(f"{fid}: neutral/missing count = {neutral}")
