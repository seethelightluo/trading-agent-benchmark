import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: volatility-normalized short reversal, with medium trend filter removed (diversifier)
D={}
for s in ASSETS:
    x=get_stock_daily_data(s, days=2300)
    if x is not None:
        x=x.sort_values('date').set_index('date'); D[s]=x['close'].astype(float)
common=sorted(set.intersection(*[set(v.index) for v in D.values()]))
P=pd.DataFrame({s:D[s].reindex(common) for s in ASSETS},index=common).ffill()
R=P.pct_change(); dates=[]; vals=[]; fwd=[]; turnovers=[]
for i in range(31,len(P)-1):
    # only completed date i, factor predicts next daily return
    rr=R.iloc[:i+1]
    f={}; fr={}
    for s in ASSETS:
        z=rr[s].iloc[-5:].sum(); v=rr[s].iloc[-20:].std()
        f[s]=-z/(v+1e-6) # reversal
        fr[s]=R[s].iloc[i+1]
    good=[s for s in ASSETS if np.isfinite(f[s]) and np.isfinite(fr[s])]
    if len(good)>=8:
        dates.append(P.index[i]); vals.append([f[s] for s in good]); fwd.append([fr[s] for s in good])
        ranks=pd.Series(f).rank(pct=True); turnovers.append(float(np.mean(np.abs(ranks-ranks.shift(1))) if len(ranks)>1 else 0))
ics=[]; nobs=[]
for a,b in zip(vals,fwd):
    q=np.corrcoef(a,b)[0,1]
    if np.isfinite(q): ics.append(q); nobs.append(len(a))
ics=np.array(ics); mu=ics.mean(); sd=ics.std(ddof=1); icir=mu/sd*np.sqrt(252) if sd>0 else 0
# equal regime halves and decay 5d
print('candidate=vol_normalized_5d_reversal dates=%d meanN=%.1f coverage=%.3f IC=%.6f ICIR=%.6f hit=%.3f turnover=%.4f'% (len(ics),np.mean(nobs),len(ics)/len(P),mu,icir,np.mean(ics>0),np.mean(turnovers)))
for label, sl in [('early',slice(None,len(ics)//2)),('late',slice(len(ics)//2,None))]:
 q=ics[sl]; print(label,'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>2 and q.std(ddof=1)>0 else 0)
for h in [1,5,10]:
 qs=[]
 for i in range(31,len(P)-h):
  f={s:-R[s].iloc[:i+1].iloc[-5:].sum()/(R[s].iloc[:i+1].iloc[-20:].std()+1e-6) for s in ASSETS}
  y={s:P[s].iloc[i+h]/P[s].iloc[i]-1 for s in ASSETS}; g=[s for s in ASSETS if np.isfinite(f[s]) and np.isfinite(y[s])]
  if len(g)>=8: qs.append(np.corrcoef([f[s] for s in g],[y[s] for s in g])[0,1])
 print('decay',h,np.nanmean(qs),len(qs))
print('period',P.index[0],P.index[-1])
# save artifacts for audit
out=pd.DataFrame({'date':dates,'ic':ics}); out.to_csv('/tmp/miner3_signal.csv',index=False)
