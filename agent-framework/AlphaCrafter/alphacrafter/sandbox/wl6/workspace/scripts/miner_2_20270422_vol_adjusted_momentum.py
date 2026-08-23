import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(D).ffill().loc[:'2027-04-21']; r=p.pct_change()
# lagged volatility-adjusted 20d trend; annualization cancels in ranks
f=(p.pct_change(20)/(r.rolling(20).std()+1e-12)).shift(1)
fr={h:p.shift(-h).div(p)-1 for h in [1,3,5,10]}
print('period',p.index.min().date(),p.index.max().date(),'assets',len(A))
for h,y in fr.items():
 vals=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 s=pd.Series(vals,index=ds).dropna(); print('H',h,'dates',len(s),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(ddof=1),5),'hit',round((s>0).mean(),4))
 for nm,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-24':('2023','2024-12-31'),'2025-26':('2025','2026-12-31'),'2027':('2027','2027-04-21')}.items():
  q=s.loc[a:b]; print(' ',nm,len(q),round(q.mean(),5) if len(q) else np.nan,round(q.mean()/q.std(ddof=1),5) if len(q)>1 else np.nan)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
