import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p,parse_dates=['date']).set_index('date'); D[s]=x['close'].astype(float)
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
def run():
 rows=[]
 for dt in P.index:
  past=P.loc[:dt]; rr=R.loc[:dt]
  if len(past)<65: continue
  # defensive-relative 20d trend: asset performance versus defensive basket
  base=rr[['XAU','US10Y','CN10Y']].iloc[-20:].sum(axis=1).median()
  f=rr.iloc[-20:].sum()-base
  f=f.replace([np.inf,-np.inf],np.nan)
  fw=P.shift(-10).loc[dt]/P.loc[dt]-1
  z=pd.concat([f,fw],axis=1).dropna()
  if len(z)>=8:
   rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
 print('IC %.5f ICIR %.5f hit %.4f turnover %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean(), np.nan))
 for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-12-31'),('2028-09-01','2029-03-26')]:
  x=q.loc[a:b].ic; print(a,b,len(x),round(x.mean(),5),round(x.mean()/x.std(ddof=1),5) if len(x)>1 else np.nan)
 q.to_csv('scripts/miner_1_20290326_defensive_relative_ic.csv')
run()
