import numpy as np,pandas as pd
from pathlib import Path
root=Path('../persistent'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-09-23')
def load(s):
 p=root/'stock_data'/(s+'.csv'); return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
pd0=pd.concat({s:load(s)['close'] for s in U},axis=1)
vol=pd.concat({s:load(s)['volume'] for s in U},axis=1).replace(0,np.nan)
r=pd0.pct_change(); ret5=pd0.pct_change(5); vs=np.log(vol/vol.rolling(20,min_periods=15).mean())
# Volume-confirmed 5d momentum: directional return weighted by abnormal contemporaneous volume.
f=ret5*vs
f=f.replace([np.inf,-np.inf],np.nan).loc[:C]
rows=[]
for h in [1,5,10]:
 y=pd0.shift(-h).div(pd0)-1; q=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 a=pd.DataFrame(q,columns=['date','ic','n']).set_index('date'); rows.append(a)
 print('h',h,'dates',len(a),'avgN',round(a.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
a=rows[0]; print('coverage',a.n.sum()/(len(a)*15),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for y0,y1 in [('2020','2022'),('2023','2024'),('2025','2026')]:
 b=a.loc[y0:y1]; print('regime',y0,y1,'dates',len(b),'IC %.6f ICIR %.6f'%(b.ic.mean(),b.ic.mean()/b.ic.std(ddof=1)))
out=[]
for d in f.index:
 for s in U:
  if pd.notna(f.loc[d,s]): out.append({'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(f.loc[d,s])})
pd.DataFrame(out).to_csv('scripts/miner_3_20260924_volume_confirmed_momentum_signal.csv',index=False)
print('cutoff',C.date(),'signal_rows',len(out),'symbols',len(U))
