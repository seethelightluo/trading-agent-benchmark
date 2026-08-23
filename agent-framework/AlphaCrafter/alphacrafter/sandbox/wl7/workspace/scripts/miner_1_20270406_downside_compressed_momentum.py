import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 df=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: df=fn(s,days=3000)
  except Exception: df=None
  if df is not None and len(df)>=120: break
 if df is None: continue
 d=df[['date','close']].copy().sort_values('date'); d['r']=d.close.pct_change()
 mom=d.close/d.close.shift(20)-1
 down=d.r.where(d.r<0,0).rolling(30).std()*np.sqrt(20)
 v20=d.r.rolling(20).std(); v60=d.r.rolling(60).std()
 fac=(mom/(down+1e-8)*(v60/(v20+1e-8)).clip(.5,2)).shift(1)
 for dt,f,fr in zip(d.date,fac,d.close.shift(-1)/d.close-1): rows.append((dt,s,f,fr))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd1']); obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8: obs.append((dt,len(g),g.factor.corr(g.fwd1,method='spearman')))
o=pd.DataFrame(obs,columns=['date','n','ic']); o.date=pd.to_datetime(o.date); mu=o.ic.mean(); sd=o.ic.std(ddof=1)
print('factor=downside_adjusted_compressed_momentum');print('dates',len(o),'avg_n',o.n.mean(),'coverage',x.factor.notna().mean(),'period',o.date.min(),o.date.max());print('daily_ic',mu,'daily_icir',mu/sd*np.sqrt(len(o)),'hit',(o.ic>0).mean())
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 q=o[(o.date.dt.year>=a)&(o.date.dt.year<=b)];print('regime',a,b,'dates',len(q),'ic',q.ic.mean() if len(q) else np.nan)
p=x.dropna(subset=['factor']).pivot(index='date',columns='symbol',values='factor');t=[]
for i in range(1,len(p)):
 a=p.iloc[i-1].rank(pct=True);b=p.iloc[i].rank(pct=True);c=a.index.intersection(b.index)
 if len(c)>=8:t.append((a[c]-b[c]).abs().mean())
print('rank_turnover',np.mean(t));x.to_csv('scripts/miner_1_20270406_downside_compressed_momentum_signal.csv',index=False)
