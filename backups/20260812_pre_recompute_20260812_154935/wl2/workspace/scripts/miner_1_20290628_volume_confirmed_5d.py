import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data

U=get_account_dict().get('watch_list',[])
if not U: U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,2600)
    if x is None or len(x)<100: x=get_index_daily_data(s,2600)
    if x is not None and len(x)>100:
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
        D[s]=x
close=pd.DataFrame({s:x['close'] for s,x in D.items()}); vol=pd.DataFrame({s:x['volume'] for s,x in D.items()})
# volume-confirmed 5d trend, information available at t, target t+1
ret=close.pct_change(5); vs=vol/vol.rolling(20,min_periods=10).median(); fac=(ret*vs.clip(0.5,2.0)).shift(1)
fwd=close.pct_change(1).shift(-1)
ics=[]; ns=[]; turns=[]
for dt in fac.index:
    a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    if dt in fac.index[1:]:
        q=fac.loc[dt].rank(pct=True); p=fac.shift(1).loc[dt].rank(pct=True)
        turns.append((q-p).abs().mean())
ic=pd.Series(ics).dropna(); print('candidate=volume_confirmed_5d_trend'); print('dates',len(ic),'avgN',np.mean(ns),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0),'turnover',np.mean(turns),'coverage',np.mean(ns)/len(U)); print('recent',ic.tail(504).mean(),ic.tail(504).mean()/ic.tail(504).std(ddof=1))
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20290628_volume_confirmed_5d_signal.csv',index=False)
