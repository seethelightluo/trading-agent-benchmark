import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in A:
 x=get_stock_daily_data(s,days=2400)
 if x is not None:D[s]=x.sort_values('date').set_index('date').close.astype(float)
common=sorted(set.intersection(*[set(x.index) for x in D.values()]));P=pd.DataFrame({s:D[s].reindex(common) for s in A},index=common).ffill();R=P.pct_change()
def f(i,s):
 r=R[s].iloc[:i+1];return r.iloc[-10:].sum()/(r.iloc[-40:].std()+1e-6)*(1 if r.iloc[-30:].sum()>=0 else -1)
z=[];ns=[];tr=[];prev=None;dates=[]
for i in range(45,len(P)-1):
 x=pd.Series({s:f(i,s) for s in A});y=R.iloc[i+1];g=x.index[x.notna()&y.notna()]
 if len(g)>=8:
  z.append(x[g].corr(y[g]));ns.append(len(g));dates.append(P.index[i]);q=x.rank(pct=True);tr.append(np.abs(q-(prev if prev is not None else q)).mean());prev=q
z=np.array(z);pd.DataFrame({'date':dates,'ic':z}).to_csv('scripts/miner_3_20290208_risk_agreement_signal.csv',index=False)
print('dates',len(z),'meanN',np.mean(ns),'period',P.index[0],P.index[-1],'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1)*np.sqrt(252),'hit',np.mean(z>0),'turnover',np.mean(tr),'coverage',len(z)/len(P))
print('late90',np.mean(z[-int(len(z)*.25):]),'late90ICIR',np.mean(z[-int(len(z)*.25):])/np.std(z[-int(len(z)*.25):],ddof=1)*np.sqrt(252))
