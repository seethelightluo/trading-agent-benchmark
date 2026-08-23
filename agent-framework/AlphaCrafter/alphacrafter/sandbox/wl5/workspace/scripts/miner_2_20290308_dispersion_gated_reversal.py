import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); rv=r.rolling(20).std()
csdisp=r.std(axis=1); med=csdisp.rolling(60).median()
# Trade only when cross-asset dispersion exceeds its trailing median; otherwise neutral.
f=(-r.rolling(5).sum()/rv).where(csdisp.gt(med),0.0)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; z=pd.concat([x,y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: rows.append((p.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z),float(csdisp.iloc[i]/med.iloc[i])))
a=pd.DataFrame(rows,columns=['date','ic','n','disp']).replace([np.inf,-np.inf],np.nan).dropna(); mu=a.ic.mean(); sd=a.ic.std(ddof=1)
print('rows',len(a),'instruments',len(U),'coverage',a.n.mean()/len(U),'IC',mu,'ICIR',mu/sd*np.sqrt(252),'hit',(a.ic>0).mean(),'turnover',(f.rank(pct=True).diff().abs().mean(axis=1)>0.05).mean())
for label,m in [('early',a.date<'2025-01-01'),('mid',(a.date>='2025-01-01')&(a.date<'2027-01-01')),('late',a.date>='2027-01-01')]:
 q=a[m]; print(label,len(q),q.ic.mean() if len(q) else np.nan)
f.to_csv('scripts/miner_2_20290308_dispersion_gated_reversal_signal.csv'); print('period',a.date.min(),a.date.max())
