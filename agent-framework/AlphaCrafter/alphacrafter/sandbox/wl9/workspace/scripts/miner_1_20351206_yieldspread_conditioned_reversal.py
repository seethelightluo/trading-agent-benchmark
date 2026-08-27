import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index()
r=p.pct_change()
# Yield-series prices are synthetic tradable yield benchmarks; spread level is a macro regime observable.
y10=pd.read_csv('../persistent/stock_data/US10Y.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
c10=pd.read_csv('../persistent/stock_data/CN10Y.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
spread=c10-y10
z=(spread-spread.rolling(60).mean())/(spread.rolling(60).std()+1e-12)
base=-r.rolling(20).sum()/(r.rolling(20).std()*np.sqrt(20)+.05)
# Strongly positive CN-US yield spread favors defensive/China-sensitive mean reversion asymmetrically.
sig=base.shift(1).mul((1+.30*np.clip(z,-1.5,1.5)).shift(1),axis=0)
ics=[];ns=[];turns=[]
for i,t in enumerate(p.index):
 x=sig.loc[t]; y=(p.shift(-10)/p-1).loc[t]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
  if i>=10:
   q=sig.iloc[i-10]; oo=q.notna()&x.notna()
   if oo.sum()>=8: turns.append(np.mean(abs(x[oo].rank()-q[oo].rank()))/oo.sum())
a=np.array(ics)
print('dates',len(a),'avgN',np.mean(ns),'minN',min(ns),'coverage',np.mean(np.array(ns)/15))
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(turns))
for h in [5,10,20,40,60]:
 y=p.shift(-h)/p-1; zc=[]
 for t in p.index:
  x=sig.loc[t]; yy=y.loc[t]; ok=x.notna()&yy.notna()
  if ok.sum()>=8: zc.append(spearmanr(x[ok],yy[ok]).statistic)
 print('decay',h,np.mean(zc),len(zc))
for lo,hi in [(2024,2026),(2027,2029),(2030,2032),(2033,2035)]:
 zc=[]
 for t in p.index:
  if lo<=t.year<=hi:
   x=sig.loc[t]; yy=(p.shift(-10)/p-1).loc[t]; ok=x.notna()&yy.notna()
   if ok.sum()>=8: zc.append(spearmanr(x[ok],yy[ok]).statistic)
 print('regime',lo,hi,np.mean(zc) if zc else None,len(zc))
pd.DataFrame(sig,columns=A).rename_axis('date').to_csv('scripts/miner_1_20351206_yieldspread_conditioned_reversal_signal.csv')
