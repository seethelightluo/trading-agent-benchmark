import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); frames[s]=d.set_index('date').sort_index()
print('assets',len(frames),'lengths',{s:len(d) for s,d in frames.items()})
def signal(c):
 r=c.pct_change(); down=r.where(r<0,0.0)
 # downside-risk adjusted medium trend; shift prevents using current close in decision
 return (c.pct_change(30)/(down.pow(2).rolling(30).mean().pow(.5)*np.sqrt(252)+.01)).shift(1)
def eval_h(h):
 rows=[]
 for s,d in frames.items():
  c=pd.to_numeric(d.close,errors='coerce'); f=signal(c); y=c.shift(-h)/c-1
  rows += [(dt,s,a,b) for dt,a,b in zip(c.index,f,y) if pd.notna(a) and pd.notna(b)]
 x=pd.DataFrame(rows,columns=['date','asset','f','y']); vals=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append((dt,g.f.corr(g.y,method='spearman')))
 ic=pd.Series(dict(vals)).dropna(); return x,ic
x,ic=eval_h(1)
print('dates',len(ic),'avg_n',x.groupby('date').size().mean(),'coverage',len(x)/sum(len(d) for d in frames.values()))
for h in [1,5,10,20]:
 _,a=eval_h(h); print('horizon',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-07-07')]:
 q=ic[(ic.index>=lo)&(ic.index<=hi)]; print('regime',label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
# rank turnover on consecutive common dates
wide=x.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); common=wide.dropna(how='all');
turn=[]
for a,b in zip(common.index[:-1],common.index[1:]):
 z=pd.concat([wide.loc[a],wide.loc[b]],axis=1).dropna();
 if len(z)>=8: turn.append((z.iloc[:,0].rank()-z.iloc[:,1].rank()).abs().mean()/max(1,len(z)))
print('rank_turnover',np.mean(turn) if turn else np.nan)
# save reproducible artifact
x.to_csv('scripts/miner_3_20270707_downside_trend_signal.csv',index=False)
print('artifact','scripts/miner_3_20270707_downside_trend_signal.csv')
