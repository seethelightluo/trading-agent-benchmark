import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,days=2800)
 if d is not None and len(d)>100:d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff();m=r.rolling(3).sum();rv=r.rolling(60).std()
# Asymmetric crash rebound: only negative 3-session shocks receive a nonlinear reversal score.
f=( (-m).clip(lower=0) * ((-m).clip(lower=0)/(rv*np.sqrt(3))) ).replace([np.inf,-np.inf],np.nan);f=f.sub(f.mean(axis=1),axis=0)
def calc(h):
 y=np.log(p.shift(-h)/p);a=[]
 for dt in f.index:
  x=f.loc[dt];q=y.loc[dt];ok=x.notna()&q.notna()
  if ok.sum()>=8:a.append((dt,x[ok].corr(q[ok],method='spearman'),ok.sum()))
 return pd.DataFrame(a,columns=['date','ic','n']).dropna()
for h in [1,3,5,10]:
 a=calc(h);print('h',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
a=calc(1)
for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2029),(2030,2030)]:
 q=a[(a.date.dt.year>=lo)&(a.date.dt.year<=hi)];print('regime',lo,hi,'dates',len(q),'IC',round(q.ic.mean(),6) if len(q) else None,'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else None)
print('coverage',f.notna().sum().sum()/(len(f)*15),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300221_downshock_signal.csv',index=False)
