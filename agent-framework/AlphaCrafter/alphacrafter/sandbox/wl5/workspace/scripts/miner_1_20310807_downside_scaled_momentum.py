import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
px={}
for s in U:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].replace(0,np.nan)
p=pd.DataFrame(px).sort_index()
r=p.pct_change()
# candidate: 20-day return scaled by downside volatility, causal; mild recent trend confirmation
mom=p.pct_change(20)
down=r.where(r<0).rolling(40,min_periods=15).std()
fac=mom/(down*np.sqrt(252)+1e-8)
# suppress unstable tails cross-sectionally, rank-compatible
fac=fac.clip(-10,10)
fwd=p.shift(-5)/p-1
rows=[]
for dt in fac.index:
 x=fac.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
ics=a.ic.dropna();
print('dates',len(a),'mean_n',a.n.mean(),'coverage',a.n.mean()/15)
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean(),'std',ics.std())
print('regimes')
for lo,hi in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-08-06')]:
 q=ics.loc[lo:hi]; print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan,(q>0).mean())
# decay
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(rr).dropna(); print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
# turnover rank signal
rank=fac.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('turnover',turn)
print('last',fac.iloc[-1].dropna().round(3).to_dict())
