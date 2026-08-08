import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in A:
 q='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(q): P[a]=pd.read_csv(q,parse_dates=['date']).set_index('date')['close']
p=pd.DataFrame(P).sort_index(); ret=np.log(p).diff()
# Short-horizon reversal scaled by robust recent absolute movement; lag one day.
# Higher signal means recent losers with orderly moves, intended to capture cross-asset mean reversion.
scale=ret.abs().rolling(20,min_periods=10).mean()
f=(-ret.rolling(3,min_periods=3).sum()/scale).shift(1)
print('DATA',p.index.min(),p.index.max(),'assets',len(P))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; out=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=pd.Series(out); print(f'h={h} dates={len(s)} meanN={np.mean(ns):.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={(s>0).mean():.4f}')
print('coverage',f.notna().stack().mean(),'mean_valid',f.notna().sum(axis=1).mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean().mean())
for label,sub in [('2024-27',f.loc['2024':'2027']),('2028-30',f.loc['2028':'2030']),('2031-34',f.loc['2031':'2034'])]:
 y=p.shift(-1)/p-1; x=[]
 for d in sub.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(x); print(label,'dates',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1))
