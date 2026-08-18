import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): P[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
# Compute on each instrument's own trading calendar, avoiding artificial NaNs.
F={}; FW={}
for s,p in P.items():
 r=p.pct_change(); ret=p.pct_change(60); down=np.sqrt((r.clip(upper=0)**2).rolling(60,min_periods=40).mean())*np.sqrt(60); dd=p/p.rolling(120,min_periods=80).max()-1
 F[s]=ret/(down+1e-8)*(1+dd.clip(-0.5,0))
 for h in [1,5,10,20]: FW.setdefault(h,{})[s]=p.pct_change(h).shift(-h)
f=pd.DataFrame(F); out={}
for h in [1,5,10,20]:
 fw=pd.DataFrame(FW[h]); vals=[]; ns=[]
 for date in f.index:
  z=pd.concat([f.loc[date],fw.loc[date]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
 q=pd.Series(vals).dropna(); ic=q.mean(); ir=ic/q.std(ddof=1)
 print(f'{h}d dates={len(q)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={ir:.6f} hit={(q>0).mean():.4f}'); out[h]=q
valid=f.notna().sum(axis=1)/len(U); rank=f.rank(axis=1,pct=True)
print(f'coverage={valid.mean():.4f} turnover={rank.diff().abs().mean(axis=1).mean()/2:.6f} total_dates={len(f)} avg_names={f.notna().sum(axis=1).mean():.2f}')
for n in [250,500]:
 q=out[5].tail(n); print(f'recent{n} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} dates={len(q)}')
print('period',f.index.min().date(),f.index.max().date())
