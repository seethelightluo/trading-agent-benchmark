import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s, days=3000)
    except Exception: x=None
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
print('loaded',len(D),list(D))
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
ret20=r.rolling(20).sum(); vol20=r.rolling(20).std(); resid=ret20.sub(ret20.median(axis=1),axis=0)
disp=r.rolling(20).std().median(axis=1); cut=disp.rolling(252,min_periods=126).quantile(.7)
stress=(ret20.median(axis=1)<0)&(disp>cut); f=(resid/vol20).where(stress,0).shift(1)
fw=np.log(p).shift(-10)-np.log(p); rows=[]
for dt in f.index:
 a=f.loc[dt]; b=fw.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),len(z)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for q,nm in [(z,'all'),(z.loc[z.index>='2025-01-01'],'recent'),(z.loc[z.index>='2028-01-01'],'late')]:
 print(nm,'dates',len(q),'avg_n',q.n.mean() if len(q) else 0,'IC',q.ic.mean() if len(q) else np.nan,'ICIR',q.ic.mean()/q.ic.std() if len(q)>1 else np.nan,'hit',(q.ic>0).mean() if len(q) else np.nan)
print('coverage',f.notna().mean().mean(),'stress_frac',stress.mean()); print('decay')
for h in [1,5,20,40]:
 fw2=np.log(p).shift(-h)-np.log(p); rr=[]
 for dt in f.index:
  zz=pd.concat([f.loc[dt],fw2.loc[dt]],axis=1).dropna()
  if len(zz)>=8 and zz.iloc[:,0].nunique()>1 and zz.iloc[:,1].nunique()>1: rr.append(zz.iloc[:,0].rank().corr(zz.iloc[:,1].rank()))
 print(h,np.mean(rr),len(rr))
if len(z): print('blocks',z.groupby(np.arange(len(z))//400).ic.mean().to_dict())
