import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);p[s]=d[d.date<='2029-08-08'].set_index('date').close
p=pd.DataFrame(p).sort_index();r=p.pct_change();D=p.pct_change(60)[['XAU','US10Y','CN10Y']].mean(axis=1);raw=-(p.pct_change(60).sub(D,axis=0))/(r.rolling(20).std()*np.sqrt(252));sig=raw.rank(axis=1,pct=True)
for h in [10,20]:
 fr=p.shift(-h)/p-1;vals=[];ds=[];ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(c):vals.append(c);ds.append(dt);ns.append(len(a))
 x=pd.Series(vals,index=ds);print('H',h,'dates',len(x),'avg_n',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(),'hit',(x>0).mean(),'coverage',np.mean(np.array(ns)/15))
 for a,b in [('2027-01-01','2028-12-31'),('2028-08-01','2029-08-08')]:
  z=x.loc[a:b];print('REG',a,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std() if len(z)>1 else np.nan)
print('turnover',sig.diff().abs().mean(axis=1).mean())
