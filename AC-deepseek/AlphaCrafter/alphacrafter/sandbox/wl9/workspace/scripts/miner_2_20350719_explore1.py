"""miner_2 exploration 2035-07-19.
Survey cross-sectional predictive power of several simple, interpretable
candidate factors across the 15-asset universe, visible data through 2035-07-18.
Reports IC/ICIR/hit/coverage + admission gate pass/fail.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "scripts")
from miner_2_lib import build, cols, fwd_panel, ev, ADMIT_IC, ADMIT_ICIR

END = pd.Timestamp("2035-07-18")
df = build(END)
c = cols(df, "close")
vol = c.pct_change().rolling(20).std() / c.rolling(20).mean()
ret = c.pct_change()

def rollbeta(asset_ret, macro_chg, w):
    return asset_ret.rolling(w).cov(macro_chg)/macro_chg.rolling(w).var()

dVIX = df["VIX__close"].pct_change()

cands = {}
cands["mom20d_skip5"] = c.shift(5)/c.shift(25)-1.0
cands["mom60d_skip5"] = c.shift(5)/c.shift(65)-1.0
cands["mom60_volscaled"] = (c.shift(5)/c.shift(65)-1.0)/(vol+1e-9)
cands["vol_z20"] = (vol-vol.rolling(60).mean())/vol.rolling(60).std()
cands["updown_ratio_20"] = ret.rolling(20).apply(lambda x:(x>0).sum()/((x<0).sum()+1e-9), raw=True)
cands["dist_high20"] = c.rolling(20).max()/c - 1.0
cands["downside_share20"] = ret.clip(upper=0).rolling(20).std()/ret.rolling(20).std()
betas={a:roll(c[a].pct_change(), dVIX, 20) for a in c.columns}
cands["beta_vix20"] = pd.DataFrame(betas)
mom20=c.shift(5)/c.shift(25)-1.0
cands["breadth20"] = (mom20>0).rolling(20).mean()

fp = fwd_panel(c, 10)
print(f"{'factor':18s} ic        icir     hit    n_ic cov_ge8 turn10 => gate")
for nm,pan in cands.items():
    e=ev(pan, fp, mv=8, min_n=20)
    turo=turnover10(pan)
    gate="PASS" if (abs(e['ic'])>=ADMIT_IC and abs(e['icir'])>=ADMIT_ICIR) else "fail"
    print(f"{nm:18s} {e['ic']:+.4f}  {e['icir']:+.3f}  {e['hit']:.3f}  {e['n_ic']:5d}  {e['cov_date_ge8']:.2f}  {turo:5.2f} => {gate}")