import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
# aligned cross asset prices and returns; macro-conditioned risk adjusted 20d momentum
px=pd.concat({s:D[s]['close'] for s in U},axis=1).sort_index(); r=px.pct_change(); vol=r.rolling(20,min_periods=15).std(); mom=px.pct_change(20)
# low-volatility environment favors continuation; high VIX reverses the momentum sign
vm=vix.reindex(px.index).ffill(); vmed=vm.rolling(60,min_periods=40).median(); regime=np.where(vm<=vmed,1.0,-1.0)
f=(mom/vol).mul(regime,axis=0)
# signal at t predicts t+1, all calculations are completed t
rows=[]; turnovers=[]
for i in range(len(px)-1):
 dt=px.index[i]; vals=f.iloc[i]; nxt=r.iloc[i+1]
 z=pd.concat([vals,nxt],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z), (z.iloc[:,0].rank(pct=True).diff().abs().mean() if False else 0)))
  # turnover proxy rank movement vs previous valid signal
  if i and f.iloc[i-1].notna().sum()>=8:
   a=f.iloc[i].rank(pct=True); b=f.iloc[i-1].rank(pct=True); turnovers.append((a-b).abs().mean())
x=pd.DataFrame(rows,columns=['date','ic','n','x']); ic=x.ic
print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/15,'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'turnover',np.nanmean(turnovers))
for lo,hi in [('2020','2022'),('2022','2024'),('2024','2026'),('2025','2026-07-16')]:
 q=x[(x.date>=lo)&(x.date<hi)].ic
 print(lo,hi,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [1,5,10]:
 rr=px.pct_change(h).shift(-h)
 a=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],rr.iloc[i]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'n',len(a),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1))
