import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 if d is not None:
  d=d[['date','close']]; d.date=pd.to_datetime(d.date)
  fs[s]=d.drop_duplicates('date').set_index('date').close
px=pd.DataFrame(fs).sort_index().ffill()
# Recovery efficiency: lagged medium-horizon return divided by trailing downside volatility.
# Positive values favor assets recovering efficiently with less downside risk.
ret=px.pct_change(60)
daily=px.pct_change()
down=daily.where(daily<0,0).rolling(60,min_periods=40).std()
sig=(ret/(down*np.sqrt(60)+1e-12)).shift(1)
ics=[]; ns=[]; tr=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],(px.shift(-10)/px-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=sig.loc[dt].dropna().rank(pct=True); p=sig.shift().loc[dt].reindex(x.index).dropna().rank(pct=True)
 if len(p): tr.append(abs(x.reindex(p.index)-p).mean())
ic=pd.Series(dict(ics))
print('dates',len(ic),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ic.mean(),'ICIR_daily_paper',ic.mean()/ic.std()*np.sqrt(252),'hit',np.mean(ic>0),'turnover',np.mean(tr))
for h in [1,5,10,20]:
 oo=[]; yy=px.shift(-h)/px-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: oo.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(oo),len(oo))
for n in [365,750,1260]:
 q=ic.tail(n);print('recent',n,q.mean(),q.mean()/q.std()*np.sqrt(252),len(q))
print('range',ic.index.min(),ic.index.max())
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20350802_recovery_efficiency_signal.csv',index=False)
pd.DataFrame({'date':ic.index,'ic':ic.values}).to_csv('scripts/miner_3_20350802_recovery_efficiency_ic.csv',index=False)
