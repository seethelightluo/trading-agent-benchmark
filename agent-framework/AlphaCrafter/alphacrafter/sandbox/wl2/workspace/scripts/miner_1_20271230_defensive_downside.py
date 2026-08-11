import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
u=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in u:
 x=get_stock_daily_data(s,days=2200)
 if x is None or len(x)<80:x=get_index_daily_data(s,days=2200)
 if x is not None and len(x):D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change()
# Defensive cross-sectional signal: favor assets with low downside volatility,
# conditioned on stressed cross-asset dispersion. Signal predicts next day.
down=r.clip(upper=0).rolling(20,min_periods=12).std()
disp=r.rolling(5,min_periods=3).std().mean(axis=1)
stress=(disp>disp.rolling(60,min_periods=20).median()).astype(float)
f=-(down.rank(axis=1,pct=True))*(0.25+0.75*stress.values[:,None])
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8:
  z.f=z.f.clip(z.f.quantile(.05),z.f.quantile(.95));rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');x=a.ic.dropna()
print('dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'period',a.index.min(),a.index.max())
for name,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-27',a.index>='2026-01-01')]:
 q=a.loc[mask].ic.dropna();print(name,len(q),q.mean(),q.mean()/q.std(ddof=1))
ss=stress.reindex(a.index).astype(bool);print('stress',ss.sum(),x[ss].mean(),'calm',x[~ss].mean())
