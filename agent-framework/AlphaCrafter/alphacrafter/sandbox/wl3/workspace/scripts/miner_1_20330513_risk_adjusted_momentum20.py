import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Price-only, broad coverage: medium momentum divided by trailing realized risk.
D={}
for s in U:
    x=get_stock_daily_data(s,days=4000)
    if x is None or len(x)==0:
        try: x=get_index_daily_data(s,days=4000)
        except FileNotFoundError: x=None
    if x is not None and len(x):
        z=x[['date','close']].copy(); z['date']=pd.to_datetime(z.date); z=z.drop_duplicates('date').set_index('date').close.astype(float)
        D[s]=z.replace([np.inf,-np.inf],np.nan)
px=pd.DataFrame(D).sort_index()
logr=np.log(px).diff()
# Signal is available at end of date t; forward return starts t+1.
ret20=px.pct_change(20)
vol20=logr.rolling(20,min_periods=15).std()*np.sqrt(252)
sig=(ret20/vol20).shift(0)
fwd=px.shift(-10)/px-1
rows=[]; vals=[]
for dt in sig.index:
    a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
    if len(a)>=8:
        ic=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
        if pd.notna(ic): rows.append((dt,ic,len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=risk_adjusted_momentum20; dates=%d avg_n=%.2f coverage=%.3f'%(len(r),r.n.mean(),r.n.sum()/(len(r)*len(U))))
print('IC=%.6f ICIR=%.6f hit=%.3f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
for h in [1,3,5,10,20]:
 f=px.shift(-h)/px-1; rr=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): rr.append(q)
 print('decay_%d IC %.6f n %d'%(h,np.mean(rr),len(rr)))
# recent windows
for n in [120,252,756]:
 q=r.tail(n).ic; print('recent',n,'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
# turnover: rank signal changes, normalized absolute signal movement
rank=sig.rank(axis=1,pct=True); print('turnover=%.6f'%rank.diff().abs().mean(axis=1).mean())
# artifact for audit/reproducibility
out=sig.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20330513_risk_adjusted_momentum20_signal.csv')
r.to_csv('scripts/miner_1_20330513_risk_adjusted_momentum20_ic.csv')
