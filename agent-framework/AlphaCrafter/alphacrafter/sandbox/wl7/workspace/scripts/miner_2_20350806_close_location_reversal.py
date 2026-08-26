import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
    try: d=get_index_daily_data(s,4200)
    except Exception:
        try:d=get_stock_daily_data(s,4200)
        except Exception:continue
    if d is not None and len(d)>100:
        d=d.copy();d['date']=pd.to_datetime(d['date']);d=d.set_index('date').sort_index();P[s]=d
cl=pd.DataFrame({s:d.close for s,d in P.items()}); hi=pd.DataFrame({s:d.high for s,d in P.items()}); lo=pd.DataFrame({s:d.low for s,d in P.items()})
ret=cl.pct_change(); rv=ret.rolling(20).std(); rng=(hi-lo)/cl
loc=((2*cl-hi-lo)/(hi-lo).replace(0,np.nan)).rolling(5).mean()
# lag all inputs: signal at t uses through t, forward return t+1..t+10
sig=-(cl/cl.shift(5)-1)/(np.sqrt(5)*rv)*loc.abs()
fwd=cl.shift(-10)/cl.shift(-1)-1
rows=[]; turnovers=[];prev=None
for dt in cl.index:
    x=sig.loc[dt].replace([np.inf,-np.inf],np.nan); y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
        ranks=x.rank(pct=True)
        if prev is not None: turnovers.append((ranks-prev).abs().mean())
        prev=ranks
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); valid=r.ic.dropna()
print('dates',len(valid),'avg_n',r.n.mean(),'IC',valid.mean(),'ICIR',valid.mean()/valid.std(),'hit',(valid>0).mean())
print('coverage',sig.notna().sum().sum()/sig.size,'turnover',np.nanmean(turnovers))
for n in [252,756,1260]:
 q=valid.tail(n);print('recent',n,q.mean(),q.mean()/q.std(),len(q))
for h in [1,5,10,20]:
 fy=cl.shift(-h)/cl.shift(-1)-1; rr=[]
 for dt in cl.index:
  z=pd.concat([sig.loc[dt],fy.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr),len(rr))
sig.to_csv('scripts/miner_2_20350806_close_location_reversal_signal.csv')
