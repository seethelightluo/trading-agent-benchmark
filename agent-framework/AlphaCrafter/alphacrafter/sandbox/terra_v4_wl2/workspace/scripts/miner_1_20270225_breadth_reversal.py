import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
# Cross-asset breadth-conditioned reversal. Breadth is lagged fraction of assets above
# their 20d average; damp reversal in broad trends and emphasize it in dispersion.
R={a:P[a].pct_change(5) for a in A}
M={a:P[a].rolling(20,min_periods=15).mean() for a in A}
rows=[]; sig=[]
for dt in sorted(set().union(*[set(x.index) for x in P.values()])):
    breadth=[]
    for a in A:
        if dt in P[a].index:
            i=P[a].index.get_loc(dt)
            if i>0 and np.isfinite(M[a].iloc[i-1]) and np.isfinite(P[a].iloc[i-1]): breadth.append(float(P[a].iloc[i-1]>M[a].iloc[i-1]))
    if len(breadth)<8: continue
    # use completed information: prior-day breadth and prior 5d return
    b=np.mean(breadth); scale=0.5+2*abs(b-0.5)
    vals={a:(-R[a].get(dt,np.nan)*scale if np.isfinite(R[a].get(dt,np.nan)) else np.nan) for a in A}
    good=[v for v in vals.values() if np.isfinite(v)]
    if len(good)<8: continue
    med=np.median(good)
    for a in A: sig.append((dt,a,vals[a]-med if np.isfinite(vals[a]) else np.nan))
    for h in [1,5,10]:
        f=[];y=[]
        for a in A:
            if dt not in P[a].index: continue
            i=P[a].index.get_loc(dt); z=vals[a]-med
            if i+h<len(P[a]) and np.isfinite(z): f.append(z); y.append(P[a].iloc[i+h]/P[a].iloc[i]-1)
        if len(f)>=8:
            q=spearmanr(f,y).statistic
            if np.isfinite(q): rows.append((dt,h,q,len(f)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=d[d.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else np.nan)
w=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal'); print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_1_20270225_breadth_reversal.csv',index=False)
