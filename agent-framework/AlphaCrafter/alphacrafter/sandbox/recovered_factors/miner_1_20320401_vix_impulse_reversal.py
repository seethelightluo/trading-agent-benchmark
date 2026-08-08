import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index()
r=p.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# Buy recent losers specifically when volatility is rising: reversal magnitude gated by VIX impulse.
vshock=v.pct_change(5); gate=(vshock>vshock.rolling(60,min_periods=30).quantile(.60)).astype(float)
f=-p.pct_change(5).mul(gate,axis=0)
print('candidate: VIX-impulse gated 5d reversal; dates',len(p),'assets',len(A))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ic=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(ic); print('H',h,'dates',len(a),'IC %.6f ICIR %.6f hit %.4f n %.2f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(ns)))
for nm,ss in [('2020-23',slice('2020','2023')),('2024-27',slice('2024','2027')),('2028-30',slice('2028','2030')),('2031+',slice('2031',None))]:
 a=[]
 for d in f.loc[ss].index:
  z=pd.concat([f.loc[d],(p.shift(-1)/p-1).loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(nm,'dates',len(a),'IC %.6f ICIR %.6f'%(a.mean(),a.mean()/a.std(ddof=1)) if len(a)>1 else 'NA')
print('coverage %.4f turnover10 %.4f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff(10).abs().mean().mean()))
# active gate frequency
print('gate frequency',gate.mean())
