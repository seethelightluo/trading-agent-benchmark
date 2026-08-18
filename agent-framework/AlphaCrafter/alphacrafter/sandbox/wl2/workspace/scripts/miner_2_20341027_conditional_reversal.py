import numpy as np, pandas as pd
ASOF=pd.Timestamp('2034-10-27'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); px[s]=d[d.date<=ASOF].set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date']); vcol='close' if 'close' in vix else vix.columns[-1]; v=vix[vix.date<=ASOF].set_index('date')[vcol].astype(float).reindex(P.index).ffill()
ret=P.pct_change(); F=-(P.shift(1)/P.shift(21)-1)/(ret.shift(1).rolling(20).std()*np.sqrt(252)); active=(v.shift(1)>v.shift(1).rolling(252,min_periods=120).median()).astype(float); F=F.mul(active,axis=0); F.index.name='date'; F.to_csv('../persistent/miner_2_20341027_vix_high_reversal_signal.csv')
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; vals=[]; counts=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and active.loc[dt]>0: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); counts.append(len(z))
 a=np.array(vals); print('horizon',h,'dates',len(a),'avg_n',np.mean(counts),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
print('coverage',F.notna().mean().mean(),'active_frac',active.mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean(),'period',P.index.min(),P.index.max(),'assets',len(P.columns))
