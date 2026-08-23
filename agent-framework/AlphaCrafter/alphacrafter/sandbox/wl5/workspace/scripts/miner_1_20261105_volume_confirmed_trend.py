import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2026-11-04'
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')
 r=x.close.pct_change()
 # volume-confirmed trend: 20d return, amplified by persistent abnormal volume; strictly lagged
 vr=(x.volume.rolling(20,min_periods=15).mean()/(x.volume.rolling(60,min_periods=40).mean()+1e-12)).clip(.25,4)
 sig=(x.close.pct_change(20)*np.log(vr)).replace([np.inf,-np.inf],np.nan)
 D[s]=pd.DataFrame({'sig':sig,'r1':x.close.pct_change().shift(-1),'r5':x.close.pct_change(5).shift(-5)})
dates=sorted(set().union(*[set(v.index) for v in D.values()]))
for h in ['r1','r5']:
 vals=[]; ds=[]; ns=[]
 for dt in dates:
  z=pd.DataFrame({s:[D[s].at[dt,'sig'] if dt in D[s].index else np.nan,D[s].at[dt,h] if dt in D[s].index else np.nan] for s in U}).T.dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 a=pd.Series(vals,index=pd.to_datetime(ds)); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=a[[lo<=d.year<=hi for d in a.index]]; print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
S=pd.DataFrame({s:D[s].sig for s in U}); print('coverage',round(S.notna().mean().mean(),4),'turnover',round(S.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'period',S.index.min().date(),S.index.max().date())
# signal artifact for gate provenance
S.rename_axis('date').stack().rename('signal').rename_axis(index=['date','symbol']).reset_index().to_csv('scripts/miner_1_20261105_volume_confirmed_trend_signal.csv',index=False)
