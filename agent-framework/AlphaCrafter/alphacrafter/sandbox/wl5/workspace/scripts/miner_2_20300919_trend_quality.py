import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# medium-horizon trend quality: 60d return normalized by 60d vol, gated by fraction of positive daily returns
xs={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is None or len(d)<200: continue
    d=d[['date','close','pct_change']].copy().set_index('date')
    xs[s]=d
close=pd.DataFrame({s:x.close for s,x in xs.items()}).sort_index()
ret=close.pct_change()
# causal signal at t, forward return from t close to t+10 close
trend=close/close.shift(60)-1
vol=ret.rolling(60,min_periods=45).std()*np.sqrt(252)
cons=(ret.gt(0).rolling(30,min_periods=20).mean()-0.5)*2
fac=trend/vol*cons
fwd=close.shift(-10)/close-1
rows=[]
for dt in fac.index:
    a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
print('dates',len(r),'meanN',round(r.n.mean(),3),'coverage',round(r.n.mean()/15,4))
for label, q in [('all',r),('2020-24',r.loc[:'2024-12-31']),('2025-27',r.loc['2025-01-01':'2027-12-31']),('2028-29',r.loc['2028-01-01':'2029-12-31']),('2030YTD',r.loc['2030-01-01':])]:
    if len(q): print(label,'N',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
# turnover based on rank ordering changes
ranks=fac.rank(axis=1,pct=True)
print('turnover',round(ranks.diff().abs().mean(axis=1).dropna().mean(),6))
print('decay')
for h in [5,10,20]:
    fw=close.shift(-h)/close-1; vals=[]
    for dt in fac.index:
      z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
      if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    vals=pd.Series(vals).dropna(); print(h,len(vals),round(vals.mean(),6),round(vals.mean()/vals.std(ddof=1),6))
# signal artifact
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20300919_trend_quality_signal.csv',index=False)
