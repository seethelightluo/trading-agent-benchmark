import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2035-12-21')
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[a]=d
p=pd.DataFrame(px).sort_index().loc[:end]; r=p.pct_change();
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill(); dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill()
mom=r.rolling(20,min_periods=15).sum().shift(1); vixchg=vix.pct_change(10).shift(1); dxychg=dxy.pct_change(10).shift(1); fwd=p.shift(-10)/p-1
factors={'vix_falling_mom':mom.mul(1-vixchg.clip(-.5,.5),axis=0), 'vix_riskoff_mom':mom.mul(1+vixchg.clip(-.5,.5),axis=0), 'dxy_confirmed_mom':mom.mul(1-dxychg.clip(-.2,.2),axis=0)}
for name,f in factors.items():
 vals=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); recent=q.tail(504)
 print(name,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4),'recentIC',round(recent.ic.mean(),6),'recentIR',round(recent.ic.mean()/recent.ic.std(ddof=1),6))
 print('regimes',*[round(x,5) for x in [q.ic.iloc[:len(q)//3].mean(),q.ic.iloc[len(q)//3:2*len(q)//3].mean(),q.ic.iloc[2*len(q)//3:].mean()]])
