import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in ASSETS:
    x=get_stock_daily_data(s,days=2300)
    if x is not None:
        x=x.sort_values('date').set_index('date'); D[s]=x.close.astype(float)
common=sorted(set.intersection(*[set(v.index) for v in D.values()]))
P=pd.DataFrame({s:D[s].reindex(common) for s in ASSETS},index=common).ffill(); R=P.pct_change()
# Risk-adjusted acceleration: recent 10d momentum relative to 40d volatility,
# retained only when its sign agrees with the prior 30d trend; causal completed bars.
def factor(i,s):
    r=R[s].iloc[:i+1]
    return (r.iloc[-10:].sum()/(r.iloc[-40:].std()+1e-6)) * (1.0 if r.iloc[-30:].sum()>=0 else -1.0)
ics=[]; ns=[]; turns=[]; dates=[]
prev=None
for i in range(45,len(P)-1):
    f=pd.Series({s:factor(i,s) for s in ASSETS}); y=R.iloc[i+1]; g=f.index[f.notna() & y.notna()]
    if len(g)>=8:
        ics.append(f[g].corr(y[g])); ns.append(len(g)); dates.append(P.index[i])
        rr=f.rank(pct=True); turns.append(np.abs(rr-(prev if prev is not None else rr)).mean()); prev=rr
z=np.asarray(ics); mu=np.nanmean(z); sd=np.nanstd(z,ddof=1)
print('dates',len(z),'meanN',np.mean(ns),'coverage',len(z)/len(P),'IC',mu,'ICIR',mu/sd*np.sqrt(252),'hit',np.mean(z>0),'turnover',np.mean(turns))
for name,a in [('early',z[:len(z)//2]),('late',z[len(z)//2:])]: print(name,'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1)*np.sqrt(252),'hit',np.mean(a>0))
for h in [1,5,10]:
 q=[]
 for i in range(45,len(P)-h):
  f=pd.Series({s:factor(i,s) for s in ASSETS}); y=P.iloc[i+h]/P.iloc[i]-1; g=f.index[f.notna() & y.notna()]
  if len(g)>=8:q.append(f[g].corr(y[g]))
 print('decay',h,'IC',np.nanmean(q),'dates',len(q))
pd.DataFrame({'date':dates,'ic':z}).to_csv('scripts/miner_3_20290125_risk_agreement_signal.csv',index=False)
print('period',P.index[0],P.index[-1],'assets',len(ASSETS))
