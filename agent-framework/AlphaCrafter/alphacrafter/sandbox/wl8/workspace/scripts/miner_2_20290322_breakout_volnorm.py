import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d['close']
cl=pd.DataFrame(px).sort_index(); r=cl.pct_change()
# Trend persistence: signed distance from 60d low/high, scaled by recent realized volatility.
hi=cl.rolling(60,min_periods=40).max(); lo=cl.rolling(60,min_periods=40).min()
mid=(hi+lo)/2; span=(hi-lo).replace(0,np.nan)
signed=(cl-mid)/span
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
sig=(signed/vol).replace([np.inf,-np.inf],np.nan)
rows=[]; horizons=[1,3,5,10]
for h in horizons:
 fr=cl.shift(-h)/cl-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): rr.append((dt,q,len(z)))
 D=pd.DataFrame(rr,columns=['date','ic','n']).set_index('date')
 for label,sub in [('full',D),('recent180',D.tail(180)),('recent360',D.tail(360))]:
  a=sub.ic; print(h,label,'dates',len(a),'avg_n',round(sub.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==5: D5=D
print('coverage',round(sig.notna().sum(axis=1).mean()/len(U),4),'period',D5.index.min().date(),D5.index.max().date())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20290322_breakout_volnorm_signal.csv',index=False)
print('regimes5d')
for label,sub in [('2020-22',D5.loc['2020':'2022']),('2023-25',D5.loc['2023':'2025']),('2026',D5.loc['2026']),('2027-29',D5.loc['2027':])]:
 a=sub.ic; print(label,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6) if len(a)>1 else np.nan)
