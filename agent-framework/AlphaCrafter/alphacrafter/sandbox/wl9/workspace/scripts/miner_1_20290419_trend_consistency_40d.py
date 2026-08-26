import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# candidate: 40d trend weighted by directional consistency and penalized downside volatility
frames={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=x.close.pct_change()
R=pd.DataFrame(frames).sort_index(); R=(1+R).cumprod() # placeholder
# rebuild from aligned prices: fill non-trading observations, then returns
R0=pd.DataFrame({s:get_stock_daily_data(s,days=4000).set_index('date')['close'] for s in U}).sort_index().ffill()
R=R0.pct_change()
# lag-safe signal at t, forward return t+1..t+10
ret40=R.rolling(40,min_periods=35).apply(lambda x: np.prod(1+x)-1, raw=True)
pos=R.gt(0).rolling(40,min_periods=35).mean()
down=R.where(R<0,0).pow(2).rolling(40,min_periods=35).mean().pow(.5)
f=ret40*pos/(down+1e-6)
f=f.replace([np.inf,-np.inf],np.nan)
fr=(1+R).rolling(10).apply(lambda x: np.prod(1+x)-1,raw=True).shift(-10)
ics=[]; dates=[]; counts=[]
for dt in f.index:
    a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); counts.append(len(z))
ics=pd.Series(ics,index=pd.DatetimeIndex(dates)).dropna()
mean=ics.mean(); sd=ics.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
print('candidate=trend_consistency_40d horizon=10')
print('dates',len(ics),'assets_mean',np.mean(counts),'coverage',np.mean(counts)/15)
print('IC %.9f ICIR %.9f hit %.6f'%(mean,icir,(ics>0).mean()))
for name,sel in [('2020-2023',ics.index.year<=2023),('2024-2026',(ics.index.year>=2024)&(ics.index.year<=2026)),('2027-2028',(ics.index.year>=2027)&(ics.index.year<=2028)),('2029',ics.index.year==2029),('recent252',np.arange(len(ics))>=len(ics)-252)]:
 q=ics[sel]; print(name,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
print('decay')
for h in [1,5,10,20]:
 ff=(1+R).rolling(h).apply(lambda x:np.prod(1+x)-1,raw=True).shift(-h); ii=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: ii.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(ii).dropna(); print(h,q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
