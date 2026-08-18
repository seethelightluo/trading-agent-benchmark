import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2030-01-23')
pdct={}
for a in assets:
 d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 pdct[a]=d[d.index<=cutoff]
p=pd.DataFrame(pdct).sort_index()
ret20=p.pct_change(20); f=ret20.sub(ret20.median(axis=1),axis=0).shift(1)
for h in [1,5,10,20]:
 r=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],r.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=np.array(vals); print(h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/np.std(x,ddof=1)*np.sqrt(252),6),'hit',round(np.mean(x>0),4),'minN',min(ns))
 y=x[-250:];print(' recent250',round(np.mean(y),6),round(np.mean(y)/np.std(y,ddof=1)*np.sqrt(252),6))
r=f.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).dropna().mean(),'coverage',f.notna().mean().mean())
print('cutoff',cutoff.date(),'last_used',p.index.max().date())
