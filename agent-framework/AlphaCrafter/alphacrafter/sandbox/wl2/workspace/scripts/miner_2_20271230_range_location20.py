import warnings; warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
u=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in u:
 x=get_stock_daily_data(s,2200)
 if x is None or len(x)<80:x=get_index_daily_data(s,2200)
 if x is not None and len(x):D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change(); hi=p.rolling(20,min_periods=12).max();lo=p.rolling(20,min_periods=12).min()
# Rolling range-location, with a one-day lag implicit in forward test.
f=(p-lo)/(hi-lo)-.5
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8:rows.append((p.index[i],len(z),z.f.clip(z.f.quantile(.05),z.f.quantile(.95)).corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');x=a.ic.dropna();print('dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'period',a.index.min(),a.index.max())
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-27',a.index>='2026-01-01')]:
 q=a.loc[m].ic.dropna();print(nm,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [3,5,10]:
 y=p.pct_change(h).shift(-h+1);rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:rr.append(z.f.clip(z.f.quantile(.05),z.f.quantile(.95)).corr(z.y))
 print('h',h,'IC',np.nanmean(rr),'n',len(rr))
