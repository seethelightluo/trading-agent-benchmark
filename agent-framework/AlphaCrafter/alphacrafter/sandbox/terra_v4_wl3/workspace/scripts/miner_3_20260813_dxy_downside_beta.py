import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-08-12')
def load(p):
 d=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].sort_index(); return d[d.index<=end]
px={s:load('../persistent/stock_data/'+s+'.csv') for s in U}; m=load('../persistent/index_data/DXY.csv').pct_change().rename('m'); R=pd.DataFrame({s:p.pct_change() for s,p in px.items()})
rows=[]
for dt in R.index:
 if dt not in m.index: continue
 hist=R.loc[:dt].tail(41).iloc[:-1]; mh=m.loc[:dt].tail(41).iloc[:-1]; ix=hist.index.intersection(mh.index); hist=hist.loc[ix]; mh=mh.loc[ix]
 if len(hist)<30: continue
 vals={}; mask=(mh<0)
 if mask.sum()<10: continue
 for s in U:
  x=hist[s][mask].dropna(); z=mh.loc[x.index]
  if len(x)>=10 and z.var()>1e-12: vals[s]=-x.cov(z)/z.var()
 f=pd.Series(vals); nxt=R.loc[dt].reindex(f.index); ok=f.notna()&nxt.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(f[ok],nxt[ok]).statistic,f,nxt))
ics=pd.Series({d:ic for d,ic,_,_ in rows}); print('dates',len(ics),'avg_n',np.mean([len(f) for _,_,f,_ in rows]),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=ics[(ics.index>=a)&(ics.index<=b+'-12-31')]; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10]:
 z=[]
 for dt,_,f,_ in rows:
  fut=R.loc[R.index>dt].head(h).reindex(columns=f.index).add(1).prod()-1; ok=f.notna()&fut.notna()
  if ok.sum()>=8:z.append(spearmanr(f[ok],fut[ok]).statistic)
 z=pd.Series(z);print('h',h,'n',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
turn=[]; prev=None
for d,_,f,_ in rows:
 r=f.rank(pct=True)
 if prev is not None: turn.append((r-prev).abs().mean())
 prev=r
print('turnover',np.mean(turn))
