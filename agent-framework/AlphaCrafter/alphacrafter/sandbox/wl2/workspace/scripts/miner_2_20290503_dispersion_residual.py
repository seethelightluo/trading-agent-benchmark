import numpy as np,pandas as pd,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in A:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close
P=pd.concat(D,axis=1).sort_index().ffill(); R=P.pct_change().values; n=len(P); k=len(A); M=np.nanmean(R,axis=1)
# residual mean-reversion, activated only when cross-asset dispersion is high; beta and residual are trailing-window estimates
for hold in [3,5,10]:
  ic=[]; ns=[]; prev=None; turns=[]; active=0
  for i in range(130,n-1):
   disp=np.nanstd(R[max(0,i-9):i+1],axis=1).mean()
   hist=np.array([np.nanstd(R[j-9:j+1],axis=1).mean() for j in range(max(10,i-119),i+1)])
   if not np.isfinite(disp) or disp<=np.nanmedian(hist): continue
   active+=1; f=np.full(k,np.nan)
   for c in range(k):
    rr=R[i-29:i+1,c]; mm=M[i-29:i+1]; ok=np.isfinite(rr)&np.isfinite(mm)
    if ok.sum()<15: continue
    beta=np.cov(rr[ok],mm[ok],ddof=1)[0,1]/(np.var(mm[ok],ddof=1)+1e-8)
    f[c]=-np.nansum((rr-beta*mm)[-hold:])/(np.nanstd(rr)+1e-6)
   ok=np.isfinite(f)&np.isfinite(R[i+1]);
   if ok.sum()>=8:
    ic.append(spearmanr(f[ok],R[i+1,ok]).statistic); ns.append(ok.sum())
    q=pd.Series(f).rank(pct=True).values
    if prev is not None: turns.append(np.nanmean(abs(q-prev)))
    prev=q
  x=np.array(ic); print(f'hold {hold} dates {len(x)} active {active} avgN {np.mean(ns):.2f} IC {x.mean():.6f} ICIR {x.mean()/x.std(ddof=1):.6f} hit {np.mean(x>0):.4f} turnover {np.mean(turns):.4f}')
  for label,lo,hi in [('early',130,n//2),('late',n//2,n-1)]:
   a=[]
   for i in range(max(130,lo),min(hi,n-1)):
    disp=np.nanstd(R[i-9:i+1],axis=1).mean(); hist=np.array([np.nanstd(R[j-9:j+1],axis=1).mean() for j in range(max(10,i-119),i+1)])
    if disp<=np.nanmedian(hist): continue
    f=[]
    for c in range(k):
     rr=R[i-29:i+1,c]; mm=M[i-29:i+1]; ok=np.isfinite(rr)&np.isfinite(mm); beta=np.cov(rr[ok],mm[ok],ddof=1)[0,1]/(np.var(mm[ok],ddof=1)+1e-8); f.append(-np.nansum((rr-beta*mm)[-hold:])/(np.nanstd(rr)+1e-6))
    ok=np.isfinite(f)&np.isfinite(R[i+1]);
    if ok.sum()>=8:a.append(spearmanr(np.array(f)[ok],R[i+1,ok]).statistic)
   print(label,len(a),round(np.mean(a),6) if a else None)
