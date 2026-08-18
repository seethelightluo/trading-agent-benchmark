import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 p[a]=d
p=pd.DataFrame(p).sort_index().loc[:'2028-12-13']; r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
f=(-(p.pct_change(5))/(r.rolling(20).std()*np.sqrt(5))).mul((vix.pct_change(5).clip(-1,1)+1).clip(.25,1.75),axis=0).clip(-8,8)
print('assets',len(assets),'dates',len(p),'valid_dates_min8',sum(f.notna().sum(axis=1)>=8))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True,axis=1).diff().abs().mean().mean(),4))
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],p.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(vals).dropna(); recent=s.tail(250)
 print('h',h,'dates',len(s),'avgN',round(np.mean([len(pd.concat([f.iloc[i],p.pct_change(h).iloc[i+h]],axis=1).dropna()) for i in range(len(p)-h) if len(pd.concat([f.iloc[i],p.pct_change(h).iloc[i+h]],axis=1).dropna())>=8]),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(ddof=1),5),'hit',round((s>0).mean(),4),'recentIC',round(recent.mean(),5),'recentICIR',round(recent.mean()/recent.std(ddof=1),5))
print('period',p.index.min().date(),p.index.max().date())
