import sys, time
sys.path.insert(0, 'scripts')
from factor_validation_lib import load_panel, load_macro, rank_ic_series, align_fwd_returns
import pandas as pd
import numpy as np

VIS = "2035-05-22"
px = load_panel(max_date=VIS)
print("panel", px.shape, flush=True)
t0 = time.time()
ret = px.pct_change()
mom10 = px / px.shift(10) - 1
mom40 = px / px.shift(40) - 1
f = mom10.rank(axis=1, pct=True) - mom40.rank(axis=1, pct=True)
print("factor computed", round(time.time()-t0,2), flush=True)
t0 = time.time()
fwd = align_fwd_returns(px, 10)
print("fwd", round(time.time()-t0,2), flush=True)
t0 = time.time()
ic = rank_ic_series(f, fwd)
print("rank_ic_series", len(ic), round(time.time()-t0,2), flush=True)
print("ic mean", ic.mean(), flush=True)