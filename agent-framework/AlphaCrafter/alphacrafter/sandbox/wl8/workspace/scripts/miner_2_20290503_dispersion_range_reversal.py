import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2029-05-02'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d['close'].loc[:cutoff]
cl=pd.DataFrame(px).sort_index(); r=cl.pct_change()
# Cross-asset dispersion regime: activate reversal when 5d cross-sectional dispersion exceeds its trailing median.
r5=r.rolling(5,min_periods=3).sum(); csdisp=r5.std(axis=1,skipna=True); threshold=csdisp.rolling(120,min_periods=60).median(); active=(csdisp>threshold)
hi=cl.rolling(60,min_periods=40).max(); lo=cl.rolling(60,min_periods=40).min(); span=(hi-lo).replace(0,np.nan)
position=(cl-(hi+lo)/2)/span; vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
sig=(-position/vol).where(active, np.nan); results={}
for h in [1,3,5,10]:
 fr=cl.shift(-h)/cl-1; rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(ic): rows.append((dt,ic,len(z)))
 D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); results[h]=D
 for label,sub in [('full',D),('recent180',D.tail(180)),('recent360',D.tail(360))]:
  a=sub.ic; print(h,label,'dates',len(a),'avg_n',round(sub.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
D=results[5]; print('coverage',round(sig.notna().sum(axis=1).mean()/len(U),4),'activation',round(active.mean(),4),'period',D.index.min().date(),D.index.max().date())
for label,sub in [('2020-22',D.loc['2020':'2022']),('2023-25',D.loc['2023':'2025']),('2026',D.loc['2026']),('2027-29',D.loc['2027':])]:
 a=sub.ic; print('regime',label,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6) if len(a)>1 else np.nan)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20290503_dispersion_range_reversal_signal.csv',index=False)
