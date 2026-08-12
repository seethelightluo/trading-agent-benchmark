import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in A:
 x=get_stock_daily_data(s,days=2400)
 if x is not None: D[s]=x.sort_values('date').set_index('date').close.astype(float)
common=sorted(set.intersection(*[set(D[s].index) for s in A])); P=pd.DataFrame({s:D[s].reindex(common) for s in A},index=common).ffill(); R=P.pct_change(); M=R.mean(axis=1)
# Residual trend: rolling beta to cross-asset mean, residual cumulative return, scaled by residual volatility;
# continuous breadth attenuation reduces trend exposure during one-sided market stress.
def factor(i,s,look=60,volwin=20):
 r=R[s].iloc[:i+1]; m=M.iloc[:i+1]
 a=r.iloc[-look:]; b=m.iloc[-look:]
 beta=a.cov(b)/(b.var()+1e-10); e=a-beta*b
 trend=e.iloc[-20:].sum() # medium-term residual continuation
 ev=e.iloc[-volwin:].std()+1e-6
 breadth=(R.iloc[max(0,i-9):i+1]>0).mean(axis=1).mean()
 atten=1-0.7*abs(2*breadth-1) # continuous, lower in one-sided breadth
 return trend/ev*atten
for look in [40,60,90]:
 for h in [1,5]:
  ics=[];ns=[];turn=[];dates=[];prev=None
  for i in range(look+5,len(P)-h):
   f=pd.Series({s:factor(i,s,look) for s in A}); y=P.iloc[i+h]/P.iloc[i]-1; g=f.index[f.notna()&y.notna()]
   if len(g)>=8:
    ics.append(f[g].corr(y[g]));ns.append(len(g));dates.append(P.index[i]); q=f.rank(pct=True); turn.append(np.abs(q-(prev if prev is not None else q)).mean());prev=q
  z=np.array(ics); print('look',look,'h',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(np.mean(z),6),'ICIR',round(np.mean(z)/np.std(z,ddof=1)*np.sqrt(252),6),'hit',round(np.mean(z>0),4),'turn',round(np.mean(turn),4),'coverage',round(len(z)/len(P),4))
  if h==1:
   for n,a in [('early',z[:len(z)//2]),('late',z[len(z)//2:])]: print(' ',n,'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1)*np.sqrt(252),6))
   pd.DataFrame({'date':dates,'ic':z}).to_csv('scripts/miner_1_20290419_residual_trend_signal.csv',index=False)
print('period',P.index[0],P.index[-1],'assets',len(A))
