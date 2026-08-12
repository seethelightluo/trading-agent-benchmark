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
def run(window):
    ics=[]; ns=[]; rankturn=[]; dates=[]
    for i in range(31,len(P)-1):
        f={s:-R[s].iloc[:i+1].iloc[-window:].sum()/(R[s].iloc[:i+1].iloc[-20:].std()+1e-6) for s in ASSETS}
        y=R.iloc[i+1]; g=[s for s in ASSETS if np.isfinite(f[s]) and np.isfinite(y[s])]
        if len(g)>=8:
            ics.append(np.corrcoef([f[s] for s in g],[y[s] for s in g])[0,1]); ns.append(len(g)); dates.append(P.index[i])
            rankturn.append(np.mean(np.abs(pd.Series(f).rank(pct=True)-pd.Series(f).rank(pct=True).shift(1))))
    z=np.asarray(ics); mu=np.nanmean(z); sd=np.nanstd(z,ddof=1)
    print('window',window,'dates',len(z),'meanN',np.mean(ns),'coverage',len(z)/len(P),'IC',mu,'ICIR',mu/sd*np.sqrt(252),'hit',np.mean(z>0),'turnover',np.mean(rankturn))
    for name, a in [('early',z[:len(z)//2]),('late',z[len(z)//2:])]: print(name,'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1)*np.sqrt(252))
    for h in [1,5,10]:
      q=[]
      for i in range(31,len(P)-h):
        f={s:-R[s].iloc[:i+1].iloc[-window:].sum()/(R[s].iloc[:i+1].iloc[-20:].std()+1e-6) for s in ASSETS}; y=P.iloc[i+h]/P.iloc[i]-1; g=[s for s in ASSETS if np.isfinite(f[s]) and np.isfinite(y[s])]
        if len(g)>=8:q.append(np.corrcoef([f[s] for s in g],[y[s] for s in g])[0,1])
      print('decay',h,np.mean(q),len(q))
    return z,dates
z,d=run(3)
pd.DataFrame({'date':d,'ic':z}).to_csv('scripts/miner_3_20290111_short_reversal_signal.csv',index=False)
print('period',P.index[0],P.index[-1])
