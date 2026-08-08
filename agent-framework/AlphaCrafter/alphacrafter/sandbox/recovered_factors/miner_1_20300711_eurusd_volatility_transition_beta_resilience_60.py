"""One candidate: EURUSD volatility-transition beta resilience (60).
Favor assets whose EURUSD beta falls during an unusually elevated and rising EURUSD
volatility transition, relative to their normal 60-session beta. Price-only asset
returns plus observation-only EURUSD; signal is lagged one session.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']
def series(fn,s):
 d=fn(s,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:series(get_stock_daily_data,a) for a in A}).sort_index(); R=P.pct_change()
try: E=series(get_index_daily_data,'EURUSD')
except Exception: E=series(get_stock_daily_data,'EURUSD')
er=E.pct_change().reindex(P.index); ev=er.rolling(10,min_periods=7).std(); q=ev.rolling(60,min_periods=40).quantile(.75)
event=((ev>q)&(ev>ev.shift(5))).astype(float)
def beta(x,y,mask=None):
 if mask is not None: x=x.where(mask,0); y=y.where(mask,0)
 return x.rolling(60,min_periods=35).cov(y).div(y.rolling(60,min_periods=35).var().replace(0,np.nan),axis=0)
normal=beta(R,er); conditional=beta(R,er,event.astype(bool))
F=(-conditional+normal).sub((-conditional+normal).median(axis=1),axis=0).shift(1); cut=P.index.max()
def metric(h,lo=None,hi=None,sign=1):
 x=(sign*F).loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); z=[]; ns=[]
 for t in x.index:
  a=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>2:
   v=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
   if np.isfinite(v):z+=[v];ns+=[len(a)]
 if not z:return {'dates':0}
 z=np.array(z);return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),4),'mean_n':round(np.mean(ns),2),'min_n':min(ns)}
print('FACTOR eurusd_volatility_transition_beta_resilience_60 cutoff',cut.date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',F.notna().sum().sum(),'/',F.size,'coverage',round(F.notna().stack().mean(),6),'EVENT_DAYS',int(event.sum()))
for sign,name in [(1,'positive'),(-1,'inverse')]:
 print('ORIENTATION',name)
 for h in [1,5,10,20]:print('H',h,metric(h,sign=sign))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]: print('REGIME20',n,metric(20,lo,hi))
print('TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CROSS_SECTIONAL_SD',round(F.std(axis=1).mean(),6))
print('LIBRARY_CORRELATION pending full signal audit; no persistence without it.')
