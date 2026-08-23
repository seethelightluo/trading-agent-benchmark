import pandas as pd,numpy as np
from scipy.stats import spearmanr
ROOT='../persistent'; A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s,macro=False):
 p=f'{ROOT}/index_data/{s}.csv' if macro else f'{ROOT}/stock_data/{s}.csv'; x=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')['close']; return x[~x.index.duplicated()]
px=pd.concat({s:ld(s) for s in A},axis=1).sort_index(); v=ld('VIX',1).reindex(px.index).ffill(); r=px.pct_change(fill_method=None)
# Volatility-regime conditional short reversal: emphasize reversal when VIX is rising, bounded to avoid outliers.
v20=v/v.shift(20)-1; mult=1+0.75*np.tanh(v20); f=-(px/px.shift(3)-1).mul(mult,axis=0)
for h in [1,5,10]:
 q=[]
 for d in f.index:
  z=pd.concat([f.loc[d],(px.shift(-h)/px-1).loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(q,columns=['date','ic','n']).set_index('date'); print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(),5),'hit',round((q.ic>0).mean(),4))
 for y in [(2020,2022),(2023,2024),(2025,2026)]:
  x=q[(q.index.year>=y[0])&(q.index.year<=y[1])].ic; print('reg',y,round(x.mean(),5),round(x.mean()/x.std(),5),len(x))
 if h==1:
  print('coverage',round(f.notna().sum().sum()/f.size,4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
