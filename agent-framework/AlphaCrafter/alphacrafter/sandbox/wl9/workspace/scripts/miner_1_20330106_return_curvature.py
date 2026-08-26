import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index()
# long-horizon trend less recent move; all observed through t-1
f=(P.shift(1)/P.shift(61)-1)-(P.shift(1)/P.shift(11)-1)
for h in [10,20,40,60]:
 a=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.x,z.y).statistic);ns.append(len(z))
 a=np.array(a); print(h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),len(a),np.mean(ns)))
valid=f.notna().sum(axis=1)>=8
print('coverage %.4f turnover %.6f rows %d instruments %d'%(f.notna().sum().sum()/f.size,f.rank(pct=True).diff().abs().mean(axis=1).where(valid).mean(),len(P),len(U)))
