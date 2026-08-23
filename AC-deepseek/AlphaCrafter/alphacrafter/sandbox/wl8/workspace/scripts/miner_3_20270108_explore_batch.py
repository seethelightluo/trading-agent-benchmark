import sys
sys.path.insert(0,'scripts')
import numpy as np, pandas as pd
import miner_3_20260730_common as c

data = c.load_data(days=3200)
cl = c.align_panel(data)
ret = cl.pct_change()

def ic_stats(factor_map, start, end, h=10, min_assets=8):
    closes = {a: d['close'].astype(float) for a,d in data.items()}
    fdf = pd.DataFrame(factor_map)
    rdf = pd.DataFrame({a: (cc.shift(-h)/cc - 1.0) for a,cc in closes.items()})
    common = fdf.index.intersection(rdf.index)
    common = common[(common>=pd.Timestamp(start))&(common<=pd.Timestamp(end))]
    ics=[]
    for d in common:
        f=fdf.loc[d].dropna(); r=rdf.loc[d].dropna()
        both=f.index.intersection(r.index)
        if len(both)<min_assets: continue
        ic,_=c.spearmanr(f[both], r[both])
        if np.isfinite(ic): ics.append(ic)
    if len(ics)<5: return None
    a=np.array(ics)
    mean=a.mean(); sd=a.std(ddof=1) if len(a)>1 else 0
    return dict(ic=mean, icir=mean/sd if sd>0 else 0, hit=(a>0).mean(), n=len(a))

cands = {}
for a in cl:
    roll_max = cl[a].rolling(60).max()
    cands.setdefault('drawdown_20x60',{})[a] = cl[a]/roll_max - 1.0
for a in cl:
    r = ret[a]
    mom = cl[a]/cl[a].shift(20)-1.0
    vol = r.rolling(20).std()
    cands.setdefault('sharpe_mom_20',{})[a] = mom/(vol+1e-9)
for a in cl:
    cands.setdefault('skew_20',{})[a] = ret[a].rolling(20).skew()
for a in cl:
    hi=cl[a].rolling(20).max(); lo=cl[a].rolling(20).min()
    cands.setdefault('range_pos_20',{})[a] = (cl[a]-lo)/(hi-lo+1e-9)
for a in cl:
    r=ret[a]
    cands.setdefault('vol_ratio_10x60',{})[a] = r.rolling(10).std()/r.rolling(60).std()
for a in cl:
    r=ret[a]
    eff=(cl[a]-cl[a].shift(20)).abs() / (r.abs().rolling(20).sum()+1e-9)
    cands.setdefault('eff_ratio_20',{})[a]=eff

recents=[('2020-01-01','2032-01-07'),('2029-01-01','2032-01-07'),('2026-01-01','2032-01-07')]
for name,fmap in cands.items():
    print("=== FACTOR",name,"===")
    for s,e in recents:
        st=ic_stats(fmap,s,e)
        if st:
            print(f"  {s}..{e}: IC={st['ic']:.4f} ICIR={st['icir']:.4f} hit={st['hit']:.3f} n={st['n']}")