import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-12-12')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A}); r=p.pct_change()
# Trend consistency: signed fraction of up days, centered, multiplied by cumulative trend
# and normalized by realized volatility. Lag one session.
up=(r>0).rolling(20,min_periods=14).mean()-0.5
mom=p.pct_change(20); vol=r.rolling(20,min_periods=14).std()
f=(mom/vol)*up; f=f.shift(1)
def met(h):
 fw=p.shift(-h)/p-1; q=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(q));
 reg={}
 for n,mask in [('2020-24',x.index.year<=2024),('2025-27',(x.index.year>=2025)&(x.index.year<=2027)),('2028-29',x.index.year>=2028),('latest120',np.arange(len(x))>=len(x)-120)]:
  y=x[mask];reg[n]=[len(y),y.mean(),y.mean()/y.std() if y.std()>0 else np.nan,(y>0).mean()]
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return dict(horizon=h,dates=len(x),mean_n=np.mean(ns),ic=x.mean(),icir=x.mean()/x.std(),hit=(x>0).mean(),turnover=np.mean(turn),regimes=reg)
print('assets',len(A),'price_dates',len(p),'cell_coverage',f.count().sum()/f.size)
for h in [1,5,10,20]: print(json.dumps(met(h),default=float))
