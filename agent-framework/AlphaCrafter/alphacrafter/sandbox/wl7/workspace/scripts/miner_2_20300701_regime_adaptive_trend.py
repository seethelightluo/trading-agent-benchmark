import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s, days=4000)
    except Exception: x=None
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
print('loaded',len(D),list(D))
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Regime-adaptive trend: risk-adjusted 60d trend in constructive breadth;
# low-volatility defensive ranking when broad 20d trend is negative.
ret60=r.rolling(60).sum(); vol60=r.rolling(60).std(); ret20=r.rolling(20).sum(); vol20=r.rolling(20).std()
breadth=ret20.median(axis=1)
risk_on=breadth>0
f=(ret60/vol60).where(risk_on, -vol20).shift(1)
fw=np.log(p).shift(-10)-np.log(p); rows=[]
for dt in f.index:
 a=f.loc[dt]; b=fw.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),len(z)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('all dates',len(z),'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit',(z.ic>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2030-06-30')]:
 q=z.loc[a:b]; print(a,b,'dates',len(q),'IC',q.ic.mean() if len(q) else np.nan,'ICIR',q.ic.mean()/q.ic.std() if len(q)>1 else np.nan,'hit',(q.ic>0).mean() if len(q) else np.nan)
print('coverage',f.notna().mean().mean(),'risk_on_frac',risk_on.mean())
for h in [1,5,10,20,40]:
 fw2=np.log(p).shift(-h)-np.log(p); rr=[]
 for dt in f.index:
  zz=pd.concat([f.loc[dt],fw2.loc[dt]],axis=1).dropna()
  if len(zz)>=8 and zz.iloc[:,0].nunique()>1 and zz.iloc[:,1].nunique()>1: rr.append(zz.iloc[:,0].rank().corr(zz.iloc[:,1].rank()))
 print('decay',h,np.mean(rr),len(rr))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
