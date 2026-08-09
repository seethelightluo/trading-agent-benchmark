import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=p.pct_change(); mom=p.pct_change(20); breadth=(r>0).rolling(20,min_periods=15).mean(); f=mom*(breadth-0.5)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ic=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(ic);print('H',h,'dates',len(a),'IC %.6f ICIR %.6f hit %.4f n %.2f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(ns)))
for nm,ss in [('2024-27',slice('2024','2027')),('2028-30',slice('2028','2030')),('2031+',slice('2031',None))]:
 ic=[]
 for d in f.loc[ss].index:
  z=pd.concat([f.loc[d],(p.shift(-1)/p-1).loc[d]],axis=1).dropna()
  if len(z)>=8:ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(ic);print(nm,len(a),a.mean(),a.mean()/a.std(ddof=1))
print('coverage',f.notna().mean().mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean().mean())
