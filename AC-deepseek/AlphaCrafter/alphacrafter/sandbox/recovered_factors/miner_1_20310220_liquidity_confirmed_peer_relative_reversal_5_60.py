"""One candidate: liquidity-confirmed peer-relative short-term reversal.
Cross-asset five-day relative losers are expected to rebound when their own
recent volume is unusually high versus their 60-session baseline, a simple
proxy for forced/liquidation-like dislocation rather than quiet drift.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def field(a,col):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d[col],errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:field(a,'close') for a in A}).sort_index()
V=pd.DataFrame({a:field(a,'volume') for a in A}).reindex(P.index)
R=P.pct_change(); r5=R.rolling(5,min_periods=5).sum(); rel=r5.sub(r5.median(axis=1),axis=0)
# log-volume anomaly is formed only from trailing information; clip prevents a few data spikes dominating.
lv=np.log1p(V); z=(lv-lv.rolling(60,min_periods=40).mean())/lv.rolling(60,min_periods=40).std().replace(0,np.nan)
F=(-rel*z.clip(-3,3)).replace([np.inf,-np.inf],np.nan)
F=F.sub(F.median(axis=1),axis=0).shift(1); cutoff=P.index.max()
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); cs=[]; nn=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   c=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(c):cs.append(c);nn.append(len(q))
 if not cs:return {'dates':0}
 cs=np.array(cs);return {'dates':len(cs),'ic':round(float(cs.mean()),6),'icir':round(float(cs.mean()/cs.std(ddof=1)),6),'hit':round(float((cs>0).mean()),4),'mean_n':round(float(np.mean(nn)),2),'min_n':int(min(nn))}
print('FACTOR liquidity_confirmed_peer_relative_reversal_5_60 cutoff',cutoff.date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20):print('H',h,met(h))
print('REGIMES horizon5')
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]:print(n,met(5,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTION_SD',round(float(F.std(axis=1).mean()),6))
