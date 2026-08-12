import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x):break
 if x is not None and len(x):D[s]=x.assign(date=pd.to_datetime(x.date)).sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=np.log(pd.DataFrame(D).sort_index().ffill());r=p.diff();neg=r.where(r<0,0.0)
base=r.rolling(20).sum()/np.sqrt((neg**2).rolling(20).mean()).replace(0,np.nan)
v=pd.read_csv('../persistent/index_data/VIX.csv');v['date']=pd.to_datetime(v.date);v=v.set_index('date').close.astype(float).reindex(p.index).ffill();vpct=v.rolling(252,min_periods=60).rank(pct=True)
f=base.mul(1-vpct,axis=0).rolling(5,min_periods=5).mean().shift(1);fr=p.shift(-10)-p;rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8:rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('dates',len(z),'avgN',round(z.n.mean(),2),'assets',len(D),'coverage',round(z.n.mean()/len(D),4),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n);print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b];print(a,b,'n',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
f.to_csv('scripts/miner_3_20320429_vix_conditioned_momentum20_signal.csv');z.to_csv('scripts/miner_3_20320429_vix_conditioned_momentum20_ic.csv')
