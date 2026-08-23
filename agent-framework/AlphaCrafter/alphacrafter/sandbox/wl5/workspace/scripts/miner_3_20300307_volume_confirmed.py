import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d):
        d=d[['date','close','volume']].copy(); d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date')
        frames[s]=d
prices=pd.concat({s:x.close for s,x in frames.items()},axis=1).sort_index()
vol=pd.concat({s:x.volume for s,x in frames.items()},axis=1).reindex(prices.index)
# candidate: trend/vol confirmed by abnormal volume, with volume-neutral fallback
r=prices.pct_change()
vol20=r.rolling(20).std(); mom10=prices/prices.shift(10)-1
vr=vol.rolling(5).mean()/vol.rolling(60).mean()-1
# only use volume confirmation; missing/zero volume gets neutral multiplier
mult=(1+0.5*np.sign(mom10)*np.sign(vr)).where(vr.notna() & (vol>0),1.0)
f=(mom10/(vol20*np.sqrt(252)))*mult
# forward returns; causal factor at t, return t+1..t+h
for h in [5,10,20]:
    fw=prices.shift(-h)/prices-1
    ics=[]; dates=[]; ns=[]
    for dt in prices.index:
        a=f.loc[dt]; b=fw.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
    x=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
    print(h,'dates',len(x),'meanN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round((f.rank(pct=True).diff().abs().mean().mean()),6),'range',prices.index.min(),prices.index.max(),'assets',len(frames))
# regime recent
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-03-06')]:
    # repeat 10d quickly
    fw=prices.shift(-10)/prices-1; arr=[]
    for dt in prices.loc[lo:hi].index:
      z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
      if len(z)>=8: arr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    q=pd.Series(arr).dropna(); print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
# artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20300307_volume_confirmed_signal.csv',index=False)
print('artifact rows',len(out))
