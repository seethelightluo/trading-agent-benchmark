import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2028-07-16')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize()
    return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:load(s) for s in U};D={s:d for s,d in D.items() if d is not None}
rows=[]
for s,d in D.items():
 c=pd.to_numeric(d.close,errors='coerce');r=c.pct_change()
 # Acceleration in risk-adjusted trend: short trend relative to slower trend.
 # Shift one completed session before cross-sectional ranking.
 v20=r.rolling(20,min_periods=15).std()*np.sqrt(252)
 v60=r.rolling(60,min_periods=40).std()*np.sqrt(252)
 f=(c.pct_change(20)/(v20+0.02)-c.pct_change(60)/(v60+0.02)).shift(1)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.to_numpy(),
  'fr1':(c.shift(-1)/c-1).to_numpy(),'fr5':(c.shift(-5)/c-1).to_numpy(),
  'fr10':(c.shift(-10)/c-1).to_numpy(),'fr20':(c.shift(-20)/c-1).to_numpy()}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(subset=['f','fr1'])
def st(x,col):
 z=[];nn=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1:
   z.append(g.f.corr(g[col],method='spearman'));nn.append(len(g))
 z=pd.Series(z).dropna();return len(z),float(np.mean(nn)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*len(D)))
for x in ['fr1','fr5','fr10','fr20']: print(x,st(q,x))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]: print('regime',a,b,st(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr10'))
p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True)
print('turnover',float(p.diff().abs().mean().mean()))
q.to_csv('scripts/miner_3_20280717_trend_acceleration_revalidation_signal.csv',index=False)
