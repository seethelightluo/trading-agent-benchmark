import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s, days=3600)
    except Exception as e: print('missing',s); continue
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
f=-(p.pct_change(5))/(r.rolling(20).std()*np.sqrt(20)+1e-8)
f=f.clip(-5,5)
for h in [1,5,10]:
    fr=p.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
    ic=pd.Series(vals,index=pd.to_datetime(dates)).dropna(); print(f'h={h} dates={len(ic)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={ic.mean():.6f} ICIR={ic.mean()/ic.std():.6f} hit={np.mean(ic>0):.4f} turnover={f.diff().abs().mean().mean():.4f}')
    for name,start,end in [('2020-2022','2020-01-01','2022-12-31'),('2023-2024','2023-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027-2028','2027-01-01','2028-12-31'),('2029 YTD','2029-01-01','2029-12-31')]:
        q=ic.loc[start:end]; print(name, f'{q.mean():.6f}' if len(q) else 'NA', len(q))
