import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d.date=pd.to_datetime(d.date); px[a]=d.set_index('date').close
p=pd.DataFrame(px).sort_index(); p=p.loc[:'2029-06-14']
macro=pd.read_csv('../persistent/index_data/DXY.csv'); macro.date=pd.to_datetime(macro.date); dxy=macro.set_index('date').close.reindex(p.index).ffill()
r=p.pct_change()
# macro-beta residual momentum: 20d return less causal 60d beta times DXY 20d return
beta=r.rolling(60,min_periods=40).cov(dxy.pct_change()).div(dxy.pct_change().rolling(60,min_periods=40).var(),axis=0)
raw20=p.pct_change(20); resid=raw20-beta.mul(dxy.pct_change(20),axis=0)
# risk adjusted residual trend
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=resid/vol
# cross-sectional rank is signal; forward returns
for h in [5,10,20]:
 obs=[]; vals=[]; turns=[]
 for i in range(len(p)-h):
  dt=p.index[i]; nxt=p.index[i+h]
  x=f.iloc[i]; y=(p.iloc[i+h]/p.iloc[i]-1)
  ok=x.notna()&y.notna()
  if ok.sum()>=8:
   z=x[ok].rank(pct=True); obs.append(spearmanr(z,y[ok]).statistic); vals.append(ok.mean())
  if i>0:
   old=f.iloc[i-1]; ok2=x.notna()&old.notna()
   if ok2.sum()>=8: turns.append((x[ok2].rank(pct=True)-old[ok2].rank(pct=True)).abs().mean())
 a=np.array(obs); print('h',h,'dates',len(a),'assets',len(assets),'coverage',np.mean(vals),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(turns))
 for name,lo,hi in [('early','2020','2024-12-31'),('mid','2025','2026-12-31'),('late','2027','2029-06-14')]:
  q=np.array([v for dt,v in zip(p.index[:len(p)-h],obs) if str(dt)>=lo and str(dt)<=hi])
  if len(q)>2: print(name,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('usable rows',len(p),'assets',len(px),'date',p.index[-1])
