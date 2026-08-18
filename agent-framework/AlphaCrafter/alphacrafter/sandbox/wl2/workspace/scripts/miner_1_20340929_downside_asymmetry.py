import pandas as pd,numpy as np
from scipy.stats import spearmanr
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in watch}).sort_index().loc[:'2034-09-28']
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); down=r.where(r<0,0).abs().rolling(20,min_periods=15).mean(); up=r.where(r>0,0).abs().rolling(20,min_periods=15).mean()
f=((up-down)/(vol+1e-8)).shift(1)
for h in [1,3,5,10,20]:
 fr=P.shift(-h)/P-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(c): vals.append(c);ns.append(len(z));dates.append(dt)
 q=pd.Series(vals,index=dates)
 print(h,'dates',len(q),'avg_n',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 for a,b in [('2020','2025'),('2026','2028'),('2029','2031'),('2032','2034')]:
  z=q.loc[a:b];print(' ',a,b,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
