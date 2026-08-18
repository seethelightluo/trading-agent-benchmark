import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U if os.path.exists(os.path.join(base,s+'.csv'))}).sort_index()
r=P.pct_change(); breadth=r.gt(0).rolling(20,min_periods=15).mean().mean(axis=1)
# reversal strongest in broad stress, smoothly weighted by breadth distance from neutral
rev=-(P/P.shift(5)-1)
f=(rev*(1+2*(0.5-breadth).clip(-.5,.5))).shift(1)
for h in [1,3,5,10,20]:
 fr=P.shift(-h)/P-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print(f'h={h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1):.8f} hit={(a>0).mean():.4f}')
 if h==10:
  for n in [120,252,756,1260]:
   q=a[-n:];print('recent',n,'ICIR',q.mean()/q.std(ddof=1),'IC',q.mean())
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20340721_breadth_stress_reversal_signal.csv',index=False)
