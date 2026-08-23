import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
ROOT='../persistent'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym, macro=False):
 p=f'{ROOT}/index_data/{sym}.csv' if macro else f'{ROOT}/stock_data/{sym}.csv'
 x=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')['close']
 return x[~x.index.duplicated()]
px=pd.concat({s:load(s) for s in assets},axis=1).sort_index()
ret=px.pct_change(fill_method=None)
vix=load('VIX',True).reindex(px.index).ffill(); dxy=load('DXY',True).reindex(px.index).ffill()
# One interpretable idea: medium-term momentum scaled by contemporaneous macro risk appetite.
# risk appetite = negative 20d VIX change, clipped; applied only as positive multiplier, no lookahead.
base=px/px.shift(1) # unused, explicit close data
mom=px/px.shift(20)-1
vixchg=vix/vix.shift(20)-1
# smooth bounded macro multiplier: 1 - 0.5*tanh(VIX change), range .5..1.5
macro=1-0.5*np.tanh(vixchg)
factor=mom.mul(macro,axis=0)
rows=[]
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1
 vals=[]
 for dt in factor.index:
  a=factor.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5),'hit',round((q.ic>0).mean(),4))
 for p in [(2020,2022),(2023,2024),(2025,2026)]:
  x=q[(q.index.year>=p[0])&(q.index.year<=p[1])].ic
  print(' regime',p,round(x.mean(),5),round(x.mean()/x.std(ddof=1),5),len(x))
 if h==1:
  # turnover based on rank changes
  ranks=factor.rank(axis=1,pct=True); turn=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna().mean()
  print('turnover',round(turn,5),'coverage',round(factor.notna().sum().sum()/factor.size,4))
