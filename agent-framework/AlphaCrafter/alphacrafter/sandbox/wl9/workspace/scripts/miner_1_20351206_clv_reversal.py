import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in A}
idx=next(iter(D.values())).index
# Persistent close-location pressure: signed intraday range pressure, volume weighted, faded cross-sectionally.
clv=pd.DataFrame({a:(((D[a].close-D[a].low)-(D[a].high-D[a].close))/((D[a].high-D[a].low)+1e-12)).rolling(20).mean() for a in A},index=idx)
vol=pd.DataFrame({a:D[a].volume for a in A},index=idx)
vz=(vol/vol.rolling(60).median()).clip(0,4)
sig=(-clv*vz.rolling(20).mean()).shift(1)
p=pd.DataFrame({a:D[a].close for a in A},index=idx); f=p.shift(-10)/p-1
ics=[];ns=[];turn=[]
for i,t in enumerate(idx):
 x=sig.loc[t];y=f.loc[t];ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum())
  if i>=10:
   q=sig.iloc[i-10]; oo=q.notna()&x.notna()
   if oo.sum()>=8:turn.append(np.mean(abs(x[oo].rank()-q[oo].rank()))/oo.sum())
a=np.array(ics);print('dates',len(a),'avgN',np.mean(ns),'minN',min(ns),'coverage',np.mean(np.array(ns)/15));print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(turn))
for h in [5,10,20,40,60]:
 y=p.shift(-h)/p-1; z=[]
 for t in idx:
  x=sig.loc[t]; yy=y.loc[t];ok=x.notna()&yy.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],yy[ok]).statistic)
 print('decay',h,np.mean(z),len(z))
for lo,hi in [(2024,2026),(2027,2029),(2030,2032),(2033,2035)]:
 z=[]
 for t in idx:
  if lo<=t.year<=hi:
   x=sig.loc[t];yy=f.loc[t];ok=x.notna()&yy.notna()
   if ok.sum()>=8:z.append(spearmanr(x[ok],yy[ok]).statistic)
 print('regime',lo,hi,np.mean(z) if z else None,len(z))
sig.to_csv('scripts/miner_1_20351206_clv_reversal_signal.csv',index_label='date')
