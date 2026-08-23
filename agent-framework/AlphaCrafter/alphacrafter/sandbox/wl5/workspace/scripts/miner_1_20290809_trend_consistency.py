import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2029-08-08');p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);p[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(p).sort_index();r=p.pct_change(); cons=r.gt(0).rolling(20,min_periods=15).mean(); down=r.where(r<0).rolling(20,min_periods=5).std(); raw=p.pct_change(40)*cons/(down*np.sqrt(252));sig=raw.rank(axis=1,pct=True)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; vals=[];ds=[];ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(c):vals.append(c);ds.append(dt);ns.append(len(a))
 x=pd.Series(vals,index=ds);print('H',h,'dates',len(x),'avg_n',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(),'hit',(x>0).mean(),'coverage',np.mean(np.array(ns)/15))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-08-08')]:
  z=x.loc[a:b];print('REG',a,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std() if len(z)>1 else np.nan)
print('turnover',sig.diff().abs().mean(axis=1).mean());out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20290809_trend_consistency_signal.csv',index=False);print('artifact rows',len(out),'latest',out.date.max(),'symbols',out.symbol.nunique())
