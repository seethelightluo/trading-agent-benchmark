import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-08-22')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:load(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
px=pd.DataFrame({s:pd.to_numeric(d.close,errors='coerce') for s,d in D.items()}).sort_index().ffill(); r=px.pct_change()
# Residual momentum: 20-session asset return minus rolling beta to equal-weight cross-asset market return, normalized by residual volatility; one-day lag.
mkt=r.mean(axis=1); beta=r.rolling(60,min_periods=40).cov(mkt).div(mkt.rolling(60,min_periods=40).var(),axis=0)
res=r.sub(beta.mul(mkt,axis=0)); f=(res.rolling(20,min_periods=15).sum()/(res.rolling(20,min_periods=15).std()+.01)).shift(1)
rows=[]
for s in px:
 for dt in px.index:
  rows.append((dt,s,f.loc[dt,s],*(px.shift(-h).loc[dt,s]/px.loc[dt,s]-1 for h in [1,5,10,20])))
q=pd.DataFrame(rows,columns=['date','asset','f','fr1','fr5','fr10','fr20']).replace([np.inf,-np.inf],np.nan).dropna(subset=['f','fr1'])
def stat(x):
 z=[]; ns=[]
 for _,g in q.groupby('date'):
  g=g.dropna(subset=[x])
  if len(g)>=8 and g.f.nunique()>1 and g[x].nunique()>1:
   z.append(g.f.corr(g[x],method='spearman'));ns.append(len(g))
 a=pd.Series(z).dropna(); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(px.columns),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*len(px.columns)))
for h in ['fr1','fr5','fr10','fr20']: print(h,stat(h))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 old=q;q=q[(q.date.dt.year>=a)&(q.date.dt.year<=b)];print('regime',a,b,stat('fr10'));q=old
p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',p.diff().abs().mean().mean())
q.to_csv('scripts/miner_2_20270823_residual_momentum_signal.csv',index=False)
