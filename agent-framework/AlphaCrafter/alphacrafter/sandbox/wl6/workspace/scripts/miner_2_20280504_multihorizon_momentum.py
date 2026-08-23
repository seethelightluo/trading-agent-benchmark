import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(D).sort_index().loc[:'2028-05-03'].ffill(); r=p.pct_change()
parts=[r.rolling(w).sum().rank(axis=1,pct=True) for w in [20,60,120]]
raw=pd.concat(parts,axis=1,keys=['20','60','120']).T.groupby(level=1).mean().T.shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; q=[]; n=[]; ds=[]
 for d in raw.index:
  z=pd.concat([raw.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); n.append(len(z)); ds.append(d)
 s=pd.Series(q,index=ds).dropna(); print('H',h,'dates',len(s),'avg_n',round(np.mean(n),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),6),'hit',round((s>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028')]:
  x=s.loc[(s.index>=lo)&(s.index<=hi+'-12-31')]; print('REG',lo,hi,round(x.mean(),6),len(x))
print('coverage',round(raw.notna().sum(axis=1).ge(8).mean(),4),'turnover',round(raw.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
