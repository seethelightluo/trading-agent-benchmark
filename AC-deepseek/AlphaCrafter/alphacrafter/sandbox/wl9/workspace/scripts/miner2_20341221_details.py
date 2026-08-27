"""Detailed validation of candidates at horizon=10: IC, ICIR, coverage, turnover, decay, correlation."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner2_20341221_toolkit import load_panel, build_frame, compute_forward_returns, rank_ic, ASSETS, VISIBLE

np.set_printoptions(suppress=True)
panel = load_panel()
frame = build_frame(panel)

def win_skip5(p, w=60, s=5):
    return p.shift(s)/p.shift(w+s)-1.0
def drawdown(p, w=120):
    return p/p.rolling(w).max()-1.0
def winrate(p, w=20):
    return p.pct_change().rolling(w).apply(lambda x:(x>0).mean(), raw=True)
def rsi14(p):
    r=p.diff(); g=r.clip(lower=0).rolling(14).mean(); l=(-r.clip(upper=0)).rolling(14).mean()
    return g/(g+l).replace(0,np.nan)
def bsi14(p):
    r=p.diff(); u=r.clip(lower=0).rolling(14).sum(); dn=(-r.clip(upper=0)).rolling(14).sum()
    return (u-dn)/(u+dn).replace(0,np.nan)

hist = frame[frame.index <= pd.Timestamp('2034-12-20')]
cands = {}
cands['drawdown_120'] = pd.DataFrame({a: drawdown(hist[a],120) for a in hist.columns})
cands['mom_60_skip5'] = pd.DataFrame({a: win_skip5(hist[a],60,5) for a in hist.columns})
cands['winrate_20'] = pd.DataFrame({a: winrate(hist[a],20) for a in hist.columns})
cands['rsi_14'] = pd.DataFrame({a: rsi14(hist[a],14) for a in hist.columns})
cands['bsi_14'] = pd.DataFrame({a: bsi14(hist[a],14) for a in hist.columns})

# Need to align with forward returns indexed on full frame's dates
fwd10 = compute_forward_returns(frame, horizon=10)

# turnover and coverage
def turnover_rank(fdf):
    rows = fdf.dropna(how='all').rank(axis=1)
    d = rows.diff().abs().mean(axis=1)
    return d.mean()

def coverage_dates_ge8(fdf):
    good=0; tot=0
    for dt in fdf.index:
        v = fdf.loc[dt].notna().sum()
        if v>=8: good+=1
        tot+=1
    return good/tot

for name, fdf in cands.items():
    # align
    fdf = fdf.reindex(fwd.index)
    res = rank_ic(fdf, fwd, 8)
    res['factor']=name
    res['turnover_rank_20']=turnover_rank(fdf)
    res['coverage_dates_ge8']=coverage_dates_ge8(fdf)
    res['coverage_asset_days']=fdf.notna().mean().mean()
    print(f"{name:16s} IC={res['ic']: .4f} ICIR={res['icir']: .4f} hit={res['ic_hit_ratio']:.3f} ndates={res['n_ic_dates']} turn={res['turnover_rank_20']:.3f} cov={res['coverage_asset_days']:.3f} dates8={res['coverage_dates_ge8']:.3f}")