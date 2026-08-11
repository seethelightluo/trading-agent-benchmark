import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
u=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in u:
 x=get_stock_daily_data(s,days=2200)
 if x is None or len(x)<80: x=get_index_daily_data(s,days=2200)
 if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Stress-conditioned shock rebound: reverse yesterday's return, scaled by cross-asset dispersion
# Dispersion is lagged 5-day average cross-sectional volatility; all inputs are observable at t.
disp=r.rolling(5).std().mean(axis=1)
market=r.mean(axis=1).rolling(5).sum()
stress=(disp>disp.rolling(60,min_periods=20).median()).astype(float)
f=-r.shift(0).mul(stress,axis=0) # date t signal predicts t+1; zero in calm states
# avoid zero cross-section on calm dates by retain small unconditional reversal
f=-r*(0.25+0.75*stress.values[:,None])
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8:
  z.f=z.f.clip(z.f.quantile(.05),z.f.quantile(.95)); rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); x=a.ic.dropna()
print('dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'period',a.index.min(),a.index.max())
for name,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-27',a.index>='2026-01-01')]:
 q=a.loc[mask].ic.dropna(); print(name,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('stress',len(x[stress.reindex(a.index).astype(bool)]),x[stress.reindex(a.index).astype(bool)].mean(),'calm',x[~stress.reindex(a.index).astype(bool)].mean())
