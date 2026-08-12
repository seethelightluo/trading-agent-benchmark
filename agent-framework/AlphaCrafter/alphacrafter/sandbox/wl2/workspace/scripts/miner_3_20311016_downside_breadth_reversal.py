import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=3000)
    if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
    if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std()
# Broad downside-shock reversal: fade each asset's 5d risk-adjusted move
# only when the cross-asset downside breadth and median absolute move indicate stress.
breadth=(r.rolling(5).sum()<0).mean(axis=1)
shock=(breadth>0.60) & (r.rolling(5).sum().abs().median(axis=1)>0.003)
f=-(r.rolling(5).sum()/vol).where(shock,np.nan)
rows=[]
for i in range(len(p)-1):
    z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
    if len(z)>=8 and z.f.nunique()>1:
        c=z.f.corr(z.y)
        if np.isfinite(c): rows.append((p.index[i],c,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'active_dates',round(shock.mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic
 print(a,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
for h in [1,3,5,10]:
 y=p.pct_change(h).shift(-h); rr=[]
 for i in range(len(p)-h):
    z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
    if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y))
 print('decay',h,'dates',len(rr),'IC',round(np.nanmean(rr),6) if rr else None)
f.to_csv('scripts/miner_3_20311016_downside_breadth_reversal_signal.csv')
