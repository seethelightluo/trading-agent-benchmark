import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=p.pct_change()
d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill(); dz=(d-d.rolling(60).mean())/(d.rolling(60).std()+1e-12)
# Contrarian 20d return, amplified in strong-dollar stress, volatility normalized.
sig=(-r.rolling(20).sum()/(r.rolling(20).std()*np.sqrt(20)+.05)).shift(1).mul((1+.30*np.clip(dz,-1.5,1.5)).shift(1),axis=0); f=p.shift(-10)/p-1
ics=[]; ns=[]; turns=[]
for i,t in enumerate(p.index):
 x=sig.loc[t];y=f.loc[t];ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum())
  if i>=10:
   q=sig.iloc[i-10];oo=q.notna()&x.notna()
   if oo.sum()>=8:turns.append(np.mean(abs(x[oo].rank()-q[oo].rank()))/oo.sum())
a=np.array(ics);print('dates',len(a),'avgN',np.mean(ns),'minN',min(ns),'coverage',np.mean(np.array(ns)/15));print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(turns))
for h in [5,10,20,40,60]:
 y=p.shift(-h)/p-1;z=[]
 for t in p.index:
  x=sig.loc[t];yy=y.loc[t];ok=x.notna()&yy.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],yy[ok]).statistic)
 print('decay',h,np.mean(z),len(z))
for lo,hi in [(2024,2026),(2027,2029),(2030,2032),(2033,2035)]:
 z=[]
 for t in p.index:
  if lo<=t.year<=hi:
   x=sig.loc[t];y=f.loc[t];ok=x.notna()&y.notna()
   if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic)
 print('regime',lo,hi,np.mean(z) if z else None,len(z))
pd.DataFrame(sig,columns=A).rename_axis('date').to_csv('scripts/miner_1_20351122_dxy_conditioned_reversal_signal.csv')