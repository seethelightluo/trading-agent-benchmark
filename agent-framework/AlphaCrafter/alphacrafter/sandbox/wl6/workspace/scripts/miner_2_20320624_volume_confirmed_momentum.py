import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2032-06-23'
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in A}
P=pd.DataFrame({a:D[a]['close'] for a in A}).sort_index().loc[:cut]; V=pd.DataFrame({a:D[a]['volume'] for a in A}).reindex(P.index)
r=P.pct_change(); mom=P/P.shift(20)-1
# interpretable: momentum weighted by log volume surprise, cross-sectionally robust and bounded
volsur=V.div(V.rolling(60,min_periods=30).median()).replace([np.inf,-np.inf],np.nan).clip(.25,4)
F=mom*np.log(volsur)
# require volume confirmation but preserve direction; rank signal is momentum times positive volume score
F=mom*(0.5+0.5*np.log(volsur).clip(-1,1))
print('cutoff',P.index.max().date(),'dates',len(P),'assets',P.shape[1],'coverage',round(F.notna().stack().mean(),4))
for h in [5,10,20]:
 a=[]; n=[]; ds=[]; turns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.x,z.y).statistic); n.append(len(z)); ds.append(P.index[i])
   if i: turns.append(np.mean(np.sign(F.iloc[i].reindex(z.index).fillna(0).values)!=np.sign(F.iloc[i-1].reindex(z.index).fillna(0).values)))
 a=np.array(a); print({'horizon':h,'valid_dates':len(a),'avg_instruments':round(np.mean(n),3),'IC':round(float(np.mean(a)),6),'ICIR':round(float(np.mean(a)/np.std(a,ddof=1)),6),'hit_ratio':round(float(np.mean(a>0)),4),'turnover':round(float(np.mean(turns)),6)})
 if h==20: print('regimes',pd.DataFrame({'ic':a},index=ds).groupby(lambda x:x.year).ic.agg(['mean','count']).round(6).to_dict('index'))
