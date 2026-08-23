import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): D[s]=d.set_index('date')['close'].astype(float)
pd.set_option('mode.chained_assignment',None); p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); rv=r.rolling(30).std(); disp=r.std(axis=1); med=disp.rolling(90).median(); gate=((disp/med)-1).clip(0,1).fillna(0); f=(-r.rolling(10).sum()/rv).mul(gate,axis=0)
def calc(h):
 out=[]
 for i in range(len(p)-h):
  x=f.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1; z=pd.DataFrame({'x':x,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.x.std()>0 and z.y.std()>0: out.append((p.index[i],z.x.corr(z.y),len(z)))
 return pd.DataFrame(out,columns=['date','ic','n'])
a=calc(5); mu=a.ic.mean(); sd=a.ic.std(ddof=1)
print('rows',len(a),'instruments',len(U),'coverage',a.n.mean()/len(U),'IC',mu,'ICIR',mu/sd*np.sqrt(252),'hit',(a.ic>0).mean(),'turnover',(f.rank(pct=True).diff().abs().mean(axis=1)>0.05).mean())
for label,m in [('2020-24',a.date<'2025-01-01'),('2025-26',(a.date>='2025-01-01')&(a.date<'2027-01-01')),('2027+',a.date>='2027-01-01'),('2028+',a.date>='2028-01-01')]:
 q=a[m]; print(label,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
f.to_csv('scripts/miner_2_20290405_dispersion_slow_reversal_5d_signal.csv'); print('period',a.date.min(),a.date.max())
