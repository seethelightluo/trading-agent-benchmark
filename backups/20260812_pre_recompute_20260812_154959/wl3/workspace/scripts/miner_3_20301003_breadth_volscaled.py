import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None: D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); r3=r.rolling(3).sum(); v=r.rolling(20).std()*np.sqrt(20)
breadth=(r3>0).mean(axis=1)
# Fade extreme 3-day cross-asset moves, normalized by trailing asset risk; all inputs lagged by one completed day.
active=((breadth<.35)|(breadth>.65)).astype(float)
f=(-r3/v).mul(active,axis=0).replace([np.inf,-np.inf],np.nan)
rows=[]
for t in f.index:
 j=r.index.searchsorted(t,side='right')
 for h in [5,10,20]:
  k=j+h-1
  if j>=len(r) or k>=len(r): continue
  z=pd.concat([f.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
  if len(z)>=8: rows.append((t,h,z.iloc[:,0].corr(z.iloc[:,1])))
x=pd.DataFrame(rows,columns=['date','h','ic'])
print('dates',x.date.nunique(),'instruments',len(U),'coverage',round(f.notna().stack().mean(),5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for h in [5,10,20]:
 a=x[x.h==h].ic; print('H',h,'n',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030)]:
  q=x[(x.h==h)&x.date.dt.year.between(lo,hi)].ic
  print('REG',lo,hi,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'n',len(q))
# recent drift
for lo in ['2028-01-01','2029-01-01','2030-01-01']:
 q=x[(x.h==5)&(x.date>=lo)].ic; print('RECENT',lo,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),len(q))
f.to_csv('scripts/miner_3_20301003_breadth_volscaled_signal.csv')
# provenance against prior breadth signal if available
try:
 old=pd.read_csv('scripts/miner_3_20300919_breadth_reversal_signal.csv',index_col=0,parse_dates=True)
 a=f.stack().rename('new').reset_index(); b=old.stack().rename('old').reset_index(); a.columns=['date','symbol','new']; b.columns=['date','symbol','old']
 q=a.merge(b,on=['date','symbol']).dropna(); print('library_corr_breadth_raw',q.new.corr(q.old), 'pairs',len(q))
except Exception as e: print('library_corr_unavailable',e)
