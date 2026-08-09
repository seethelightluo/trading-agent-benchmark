import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 p='../persistent/stock_data/'+a+'.csv'; d=pd.read_csv(p,parse_dates=['date']).set_index('date'); return d
D={a:load(a) for a in assets}
# candidate: range-compression breakout reversal. Low recent range relative to 20d range,
# signed by 5d return; hypothesis compressed assets with positive drift continue.
cl=pd.concat({a:D[a].close for a in assets},axis=1)
ret=cl.pct_change()
range20=pd.concat({a:(D[a].high-D[a].low)/D[a].close for a in assets},axis=1).rolling(20,min_periods=15).mean()
# signal = 5d return / recent avg range, interpretable trend efficiency
fac=ret.rolling(5).sum().div(range20)
for h in [1,5,10]:
 fwd=cl.pct_change(h).shift(-h); ics=[]; dates=[]; ns=[]; ranks=[]
 for dt in fac.index:
  x=fac.loc[dt].dropna(); y=fwd.loc[dt].reindex(x.index).dropna(); x=x.reindex(y.index)
  if len(x)>=8 and x.nunique()>1 and y.nunique()>1:
   ics.append(spearmanr(x,y).statistic); dates.append(dt); ns.append(len(x))
 s=pd.Series(ics,index=dates); print('H',h,'dates',len(s),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean()))
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   z=s[(s.index.year>=lo)&(s.index.year<=hi)]; print('REG',lo,hi,len(z),z.mean(),z.mean()/z.std())
print('coverage',fac.notna().sum(axis=1).mean()/15)
# rank turnover
rr=fac.rank(axis=1,pct=True); print('turnover',rr.diff().abs().mean(axis=1).mean())
# save signal artifact
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20261008_efficiency_signal.csv',index=False)
