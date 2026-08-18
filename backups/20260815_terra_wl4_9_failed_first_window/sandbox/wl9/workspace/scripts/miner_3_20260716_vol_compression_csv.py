import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index(); r=np.log(p).diff()
# volatility compression: low recent volatility relative to medium volatility, robust rank signal
f=-(r.rolling(10,min_periods=8).std()/r.rolling(60,min_periods=40).std())
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a))
 z=np.array(vals); print('h',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1), (z>0).mean()))
q=f.rank(axis=1,pct=True); print('turnover',q.diff().abs().mean(axis=1).mean(),'coverage',f.notna().sum().sum()/f.size,'period',p.index.min(),p.index.max())
for label,a in [('early',z[:len(z)//2]),('late',z[len(z)//2:]),('recent250',z[-250:])]: print(label,a.mean(),a.mean()/a.std(ddof=1),len(a))
print('corr rev5',f.stack().corr((-p.pct_change(5)).stack()))
