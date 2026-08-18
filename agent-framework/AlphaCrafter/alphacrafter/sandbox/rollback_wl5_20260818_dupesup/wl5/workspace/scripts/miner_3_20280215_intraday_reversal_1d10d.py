import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=4000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
        D[s]=x
print('assets',len(D),{k:len(v) for k,v in D.items()})
# Intraday reversal: negative open-to-close return, demeaned cross-section.
# Signal at t uses t open/close, forward return starts t+1 close through t+10 close.
cl=pd.DataFrame({s:d['close'] for s,d in D.items()}); op=pd.DataFrame({s:d['open'] for s,d in D.items()})
intra=cl/op-1
sig=-(intra.sub(intra.median(axis=1),axis=0))
fwd=cl.shift(-10)/cl.shift(-1)-1
rows=[]
for dt in sig.index:
    a=sig.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); r=r.replace([np.inf,-np.inf],np.nan).dropna()
print('dates',len(r),'avgN',r.n.mean(),'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit', (r.ic>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31')]:
 q=r.loc[a:b].ic
 if len(q): print(a,len(q),q.mean(),q.mean()/q.std(ddof=1))
# signal turnover via rank changes, average cross-sectional rank distance
ranks=sig.rank(axis=1,pct=True); turnover=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna().mean()
print('turnover',turnover,'coverage',sig.notna().stack().mean())
out='scripts/miner_3_20280215_intraday_reversal_1d10d_signal.csv'
sig.stack().rename('signal').to_csv(out,header=True)
print('artifact',out)
