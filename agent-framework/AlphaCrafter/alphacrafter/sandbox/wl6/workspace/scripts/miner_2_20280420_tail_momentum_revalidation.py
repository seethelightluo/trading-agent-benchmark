import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(D).sort_index().loc[:'2028-04-20'].ffill();r=p.pct_change();v=r.rolling(30).std();f=(r.rolling(30).sum()/(v*np.sqrt(30))).shift(1)
for h in [1,5,10]:
 y=p.shift(-h).div(p)-1; q=[];n=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 s=pd.Series(q).dropna();print(h,len(s),round(np.mean(n),2),round(s.mean(),6),round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),6),round((s>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
