import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-01-13')
px={}
for s in U:
    d=get_stock_daily_data(s, days=3000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].drop_duplicates('date').set_index('date').close
        px[s]=x
P=pd.DataFrame(px).sort_index().ffill()
# candidate: robust distance from trailing 20-day median, scaled by trailing vol; positive means expected rebound
med=P.rolling(20,min_periods=20).median()
vol=P.pct_change().rolling(20,min_periods=20).std()*np.sqrt(252)
f=-(P/med-1)/(vol+1e-8)
fr=P.shift(-10)/P-1
rows=[]
for dt in f.index:
    if dt not in fr.index: continue
    a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# turnover proxy based on daily normalized rank changes
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('cutoff',P.index.max().date(),'dates',len(r),'instruments',len(U),'mean_n',r.n.mean(),'coverage',r.n.mean()/len(U))
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit', (r.ic>0).mean(),'turn',turn)
for h in [5,10,20]:
 ff=P.shift(-h)/P-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(rr),len(rr))
for name, sub in [('2025-26',r.loc['2025':'2026']),('2027',r.loc['2027'])]: print(name,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1) if len(sub)>1 else np.nan)
print('recent',r.tail(60).ic.mean(),len(r.tail(60)))
