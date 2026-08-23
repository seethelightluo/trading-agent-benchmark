import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    d=get_stock_daily_data(s,days=2400)
    if d is not None and len(d): D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# A slower, lower-noise reversal: 10d loss normalized by 30d volatility,
# activated continuously only when cross-asset dispersion is above its 90d median.
rv=r.rolling(30).std(); disp=r.std(axis=1); med=disp.rolling(90).median()
gate=((disp/med)-1).clip(0,1).fillna(0.0)
f=(-r.rolling(10).sum()/rv).mul(gate,axis=0)
rows=[]
for i in range(len(p)-10):
    x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1
    z=pd.concat([x,y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)>=8: rows.append((p.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).dropna(); mu=a.ic.mean(); sd=a.ic.std(ddof=1)
print('rows',len(a),'instruments',len(U),'coverage',a.n.mean()/len(U),'IC',mu,'ICIR',mu/sd*np.sqrt(252),'hit',(a.ic>0).mean(),'turnover',(f.rank(pct=True).diff().abs().mean(axis=1)>0.05).mean())
for label,m in [('2020-24',a.date<'2025-01-01'),('2025-26',(a.date>='2025-01-01')&(a.date<'2027-01-01')),('2027+',a.date>='2027-01-01'),('2028+',a.date>='2028-01-01'),('2029+',a.date>='2029-01-01')]:
 q=a[m]; print(label,len(q),q.ic.mean() if len(q) else np.nan, (q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252)) if len(q)>1 else np.nan)
for h in [5,15,20]:
 rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('horizon',h,'IC',np.nanmean(rr),'n',len(rr))
f.to_csv('scripts/miner_2_20290405_dispersion_slow_reversal_signal.csv'); print('period',a.date.min(),a.date.max())
