import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for a in A:
 x=pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date').sort_index();D[a]=x.close
p=pd.DataFrame(D).ffill().loc[:'2027-04-21']; f=p.pct_change(40).shift(1);f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10]:
 q=[];n=[]
 for d in f.index:
  z=pd.concat([f.loc[d],(p.shift(-h).div(p)-1).loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 s=pd.Series(q);print(h,len(s),np.mean(n),s.mean(),s.mean()/s.std(ddof=1)*np.sqrt(len(s)),(s>0).mean())
print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
