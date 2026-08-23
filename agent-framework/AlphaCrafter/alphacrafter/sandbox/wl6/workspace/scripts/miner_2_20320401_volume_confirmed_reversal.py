import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-03-31'); close={}; vol={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 close[s]=d.close.astype(float); vol[s]=d.volume.astype(float).replace(0,np.nan)
p=pd.concat(close,axis=1).sort_index().loc[:cut]; v=pd.concat(vol,axis=1).reindex(p.index)
r=p.pct_change(); r10=p/p.shift(10)-1
# Reversal is stronger after unusually quiet trading, avoiding raw cross-asset volume scale.
logv=np.log(v); vz=(logv-logv.rolling(60,min_periods=30).mean())/(logv.rolling(60,min_periods=30).std()+1e-12)
sig=-r10*np.clip(1-vz,-2,2)
print('candidate=volume_confirmed_reversal_10d dates=%d instruments=%d cutoff=%s'%(len(p),len(U),cut.date()))
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[]; n=[]; ds=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q); n.append(len(a)); ds.append(dt)
 z=pd.Series(z,index=ds)
 print('h=%d dates=%d avg_n=%.2f coverage=%.5f IC=%+.8f ICIR=%+.6f hit=%.4f'%(h,len(z),np.mean(n),np.mean(n)/15,z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean()))
 print('regimes',z.groupby(z.index.year).mean().round(6).to_dict())
print('turnover=%.6f'%sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
