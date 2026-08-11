import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,3000)
 D[s]=d.set_index('date').close if d is not None else pd.Series(dtype=float)
px=pd.DataFrame(D).sort_index(); r=px.pct_change(); vol=r.rolling(20,min_periods=12).std(); f=(px.pct_change(5)/vol).clip(-4,4); y=px.shift(-1).div(px)-1
z=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(q)>=8:z.append((dt,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
d=pd.DataFrame(z,columns=['date','ic','n']); m=d.ic.mean(); print({'dates':len(d),'avg_n':d.n.mean(),'IC':m,'ICIR':m/d.ic.std(ddof=1),'hit':(d.ic>0).mean(),'coverage':f.notna().sum().sum()/(f.shape[0]*len(U))})
for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
 q=d[d.date.dt.year.between(a,b)].ic; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
