"""One idea: USDCNY-risk-state conditioned short peer-relative reversal (10 sessions)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']
def series_asset(a):
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
def series_index(a):
 d=get_index_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:series_asset(a) for a in A}).sort_index(); R=P.pct_change(); med=R.median(1); rel=R.sub(med,axis=0)
fx=series_index('USDCNY').reindex(P.index).ffill(); fxr=fx.pct_change()
# Lagged continuous risk-state multiplier: elevated USD/CNY return favours reversal of recent peer losers.
z=(fxr.rolling(20,min_periods=15).mean()/fxr.rolling(60,min_periods=40).std()).clip(-3,3).shift(1)
# Cross-sectional peer-relative 10d move, reversed; multiplier preserves signal direction while attenuating benign FX regimes.
raw=-rel.rolling(10,min_periods=8).sum().shift(1)
F=raw.mul(1+z.clip(lower=0),axis=0);F=F.sub(F.median(1),axis=0)
cut=P.index.max()
def metric(h,lo=None,hi=None,sgn=1):
 x=(F*sgn).loc[lo:hi];y=(P.shift(-h)/P-1).reindex(x.index);v=[]; nn=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   k=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(k):v.append(k);nn.append(len(q))
 if not v:return {'dates':0}
 v=np.array(v);return {'dates':len(v),'ic':round(v.mean(),6),'icir':round(v.mean()/v.std(ddof=1),6),'hit':round((v>0).mean(),4),'mean_n':round(np.mean(nn),2),'min_n':min(nn)}
print('FACTOR usdcny_risk_state_conditioned_peer_reversal_10 cutoff',cut.date(),'assets',len(A),'dates',len(P))
print('CELLS',F.notna().sum().sum(),'/',F.size,'coverage',round(F.notna().stack().mean(),6))
for sg,n in [(1,'positive'),(-1,'inverse')]:
 print('ORIENTATION',n)
 for h in [1,5,10,20]:print('H',h,metric(h,sgn=sg))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_now','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]:print('REGIME10',n,metric(10,lo,hi))
print('FX_POSITIVE_STATE_DATES',int((z>0).sum()))
print('TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CROSS_SECTIONAL_SD',round(F.std(1).mean(),6))
