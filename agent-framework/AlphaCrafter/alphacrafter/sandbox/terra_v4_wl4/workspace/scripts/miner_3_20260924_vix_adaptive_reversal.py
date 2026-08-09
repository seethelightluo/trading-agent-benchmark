import numpy as np,pandas as pd
from pathlib import Path
root=Path('../persistent'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-09-23')
def load(s,macro=False):
 p=root/('index_data' if macro else 'stock_data')/(s+'.csv'); return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
p=pd.concat({s:load(s) for s in U},axis=1); v=load('VIX',True); p=p.join(v.rename('VIX'),how='inner').loc[:C]; v=p.pop('VIX'); r=p.pct_change(); vr=v.pct_change();
# Adaptive signal: stressed VIX regime favors 5d reversal; calm regime favors 20d continuation.
stress=v>v.rolling(60,min_periods=30).median(); f=pd.DataFrame(np.where(stress.values[:,None],-r.rolling(5,min_periods=5).sum().values,r.rolling(20,min_periods=15).sum().values),index=p.index,columns=p.columns)
f=f.replace([np.inf,-np.inf],np.nan); rows=[]
for h in [1,5,10]:
 y=p.shift(-h).div(p)-1; q=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 a=pd.DataFrame(q,columns=['date','ic','n']).set_index('date'); print('h',h,'dates',len(a),'avgN',a.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
 if h==1:
  print('coverage',a.n.sum()/(len(a)*15),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
  for y0,y1 in [('2020','2022'),('2023','2024'),('2025','2026')]:
   b=a.loc[y0:y1]; print('regime',y0,y1,len(b),'IC %.6f ICIR %.6f'%(b.ic.mean(),b.ic.mean()/b.ic.std(ddof=1)))
print('cutoff',C.date(),'last',f.index.max().date())
out=[]
for d in f.index:
 for s in U:
  if pd.notna(f.loc[d,s]): out.append({'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(f.loc[d,s])})
pd.DataFrame(out).to_csv('scripts/miner_3_20260924_vix_adaptive_reversal_signal.csv',index=False)
print('signal_rows',len(out))
