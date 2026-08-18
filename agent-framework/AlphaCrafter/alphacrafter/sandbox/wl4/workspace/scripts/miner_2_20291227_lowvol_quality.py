import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; end=pd.Timestamp('2029-12-27')
px={}
for a in assets:
 f=os.path.join(base,a+'.csv')
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).query('date<=@end').sort_values('date').set_index('date'); px[a]=d.close.astype(float)
p=pd.DataFrame(px).sort_index(); rr=p.pct_change();
# defensive quality: low 30d vol, low downside volatility, conditioned on positive 120d trend
v=rr.rolling(30,min_periods=20).std(); dn=rr.where(rr<0).rolling(30,min_periods=15).std(); trend=p/p.shift(120)-1
sig=(-(0.6*v+0.4*dn))*(1+0.5*trend.clip(-1,1)); sig=sig.shift(1); fwd=p.shift(-10)/p-1
ics=[]; ns=[]
for d in p.index:
 ok=sig.loc[d].notna()&fwd.loc[d].notna()
 if ok.sum()>=8: ics.append(spearmanr(sig.loc[d,ok],fwd.loc[d,ok]).statistic); ns.append(ok.sum())
ics=np.array(ics); print('dates',len(ics),'avgN',np.mean(ns),'minN',min(ns),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1)*np.sqrt(252/10),'hit',(ics>0).mean())
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; q=[]
 for d in p.index:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic)
 print('decay',h,len(q),np.mean(q))
print('coverage',sig.notna().sum(axis=1).mean()/15,'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'last',p.index.max())
