import os, json
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
    d=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: d=fn(s, days=1900)
        except Exception: pass
        if d is not None and len(d): break
    if d is None or len(d)==0: continue
    x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').sort_index()
    r=x.close.pct_change(); x['factor']=x.close.pct_change(20)/r.rolling(20,min_periods=15).std(); x['fwd5']=x.close.shift(-5)/x.close-1; x['symbol']=s
    rows.append(x.reset_index()[['date','symbol','factor','fwd5']])
if not rows: raise RuntimeError('no data')
df=pd.concat(rows,ignore_index=True); obs=[]
for dt,g in df.groupby('date'):
    g=g.replace([np.inf,-np.inf],np.nan).dropna()
    if len(g)>=8:
        ic=g.factor.corr(g.fwd5,method='spearman')
        if np.isfinite(ic): obs.append((dt,ic,len(g)))
o=pd.DataFrame(obs,columns=['date','ic','n']).sort_values('date')
print('dates',len(o),'assets',df.symbol.nunique(),'avgN',o.n.mean(),'period',o.date.min(),o.date.max())
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1), (o.ic>0).mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
 z=o[(o.date>=a)&(o.date<=b)]
 if len(z): print(a,'n',len(z),'ic %.6f icir %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)))
art=df[['date','symbol','factor']].dropna().copy(); art.to_csv('../persistent/factor_signals_miner_1_20270225_riskadj_mom20.csv',index=False)
