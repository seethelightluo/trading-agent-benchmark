import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d[d.date<='2026-07-15'].set_index('date')
D={s:load(s) for s in U}; close=pd.DataFrame({s:d.close for s,d in D.items()}); ret=close.pct_change()
# market-stress amplified one-day reversal: base reversal, increased after unusually broad moves
breadth=ret.mean(axis=1); stress=(breadth.abs()/ret.std(axis=1)).clip(0,3)
rows=[]
for s in U:
 r=ret[s]; vol=r.rolling(20,min_periods=10).std(); f=(-r/vol)*(1+0.5*stress)
 q=pd.concat([f,r.shift(-1)],axis=1); q.columns=['f','y']; q['date']=q.index; rows.append(q.reset_index(drop=True))
a=pd.concat(rows,ignore_index=True); valid=a.dropna(); obs=[]
for dt,g in valid.groupby('date'):
 g=g[(g.f!=0)&np.isfinite(g.f)&np.isfinite(g.y)]
 if len(g)>=8 and g.f.nunique()>1: obs.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
o=pd.DataFrame(obs,columns=['date','ic','n']); ic=o.ic.mean(); ir=ic/o.ic.std()
# rank turnover based on successive daily cross-sections
ranks=[]
for dt,g in valid.groupby('date'):
 if len(g)>=8: ranks.append(g.assign(rank=g.f.rank(pct=True))[['date','rank']])
rr=pd.concat(ranks); piv=rr.pivot_table(index='date',values='rank',aggfunc=list)
turn=[]
for x,y in zip(piv.index[:-1],piv.index[1:]):
 a1=dict(zip(valid[valid.date==x].index,[])) if False else None
# direct mean abs rank changes among common symbols
for dates in zip(sorted(valid.date.unique())[:-1],sorted(valid.date.unique())[1:]):
 g1=valid[valid.date==dates[0]].set_index(valid[valid.date==dates[0]].index); g2=valid[valid.date==dates[1]].set_index(valid[valid.date==dates[1]].index)
 # indices are asset-row positions, not symbols; use row order U
 v1=valid[valid.date==dates[0]].f.values; v2=valid[valid.date==dates[1]].f.values
 if len(v1)==len(U) and len(v2)==len(U): turn.append(np.mean(np.abs(pd.Series(v1).rank(pct=True).values-pd.Series(v2).rank(pct=True).values)))
print('dates',len(o),'avgN',o.n.mean(),'coverage',len(o)/valid.date.nunique(),'IC',ic,'ICIR',ir,'hit',(o.ic>0).mean(),'turn',np.mean(turn) if turn else None)
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 x=o[(o.date.dt.year>=lo)&(o.date.dt.year<=hi)].ic; print('regime',lo,hi,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std() if len(x)>1 else np.nan)
for h in [5,10]:
 rr=[]
 for s in U:
  r=ret[s]; vol=r.rolling(20,min_periods=10).std(); f=(-r/vol)*(1+0.5*stress); q=pd.concat([f,r.shift(-h).rolling(h).sum()],axis=1).dropna();q.columns=['f','y'];q['date']=q.index;rr.append(q)
 z=pd.concat(rr,ignore_index=True); vals=[spearmanr(g.f,g.y).statistic for _,g in z.groupby('date') if len(g)>=8 and g.f.nunique()>1];print('h',h,'IC',np.nanmean(vals),'ICIR',np.nanmean(vals)/np.nanstd(vals))
