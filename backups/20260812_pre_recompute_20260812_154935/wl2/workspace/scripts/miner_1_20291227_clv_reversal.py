import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ret={}; high={}; low={}; close={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None: d=get_index_daily_data(s,3000)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
 close[s]=d.close; high[s]=d.high; low[s]=d.low; ret[s]=d.close.pct_change()
C=pd.DataFrame(close); H=pd.DataFrame(high).reindex(C.index); L=pd.DataFrame(low).reindex(C.index); R=pd.DataFrame(ret).reindex(C.index)
# factor: smoothed close-location pressure, demeaned by asset's own history; low values predict rebound
loc=((C-L)/(H-L).replace(0,np.nan)).clip(0,1)
# persistent selling pressure over 10d, with volatility normalization
f=(loc.rolling(10,min_periods=7).mean()-0.5)
# reversal: negative recent location is positive score, scaled by recent vol
vol=R.rolling(20,min_periods=12).std()
f=(-f/vol.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(C)-1):
 x=f.iloc[i]; y=R.iloc[i+1]
 z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: rows.append((C.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
df=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for cutoff in [None,'2028-01-01','2029-01-01','2029-07-01']:
 q=df if cutoff is None else df.loc[cutoff:]
 ic=q.ic.mean(); sd=q.ic.std(ddof=1); print(cutoff or 'full', 'dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',ic,'ICIR',ic/sd*np.sqrt(1) if sd else np.nan,'hit', (q.ic>0).mean())
print('turnover proxy',f.rank(axis=1,pct=True).diff().abs().mean().mean())
# artifact
out=f.copy(); out.index.name='date'; out.reset_index().to_csv('scripts/miner_1_20291227_clv_reversal_signal.csv',index=False)
