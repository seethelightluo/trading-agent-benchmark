import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-02-07')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]
r=p.pct_change(); mom=r.rolling(20).sum(); vol=r.rolling(20).std()
# cross-sectional relative momentum, penalized by idiosyncratic realized volatility
f=mom.sub(mom.median(axis=1),axis=0)/(vol+1e-8)
sig=f.shift(1)
rows=[]
for h in [10,20,40]:
 y=p.shift(-h)/p-1; rows=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): rows.append((dt,q,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/(a.ic.std(ddof=1)+1e-12),6),'hit',round((a.ic>0).mean(),4))
 for name,sl in [('early',a.loc[:'2023-12-31']),('online',a.loc['2026-07-16':]),('recent',a.loc['2029-01-01':])]:
  print(name,len(sl),round(sl.ic.mean(),6),round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6))
 if h==20:
  a.to_csv('scripts/miner_1_20300207_risk_adjusted_relative_momentum_20d_signal.csv')
