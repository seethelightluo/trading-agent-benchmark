import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
 if x is not None and len(x): D[s]=x.set_index('date')
p=pd.DataFrame({k:v.close.astype(float) for k,v in D.items()}).sort_index().ffill(); r=p.pct_change()
# Trend persistence: medium return rewarded, but penalize downside volatility and unstable recent path.
down=r.where(r<0,0.0)
downvol=down.rolling(30,min_periods=15).std()
trend=r.rolling(20,min_periods=15).sum()/(downvol*np.sqrt(20)+1e-8)
# require recent trend agreement, reducing exposure to one-day reversals
agree=(r.rolling(10,min_periods=8).mean()>0).astype(float).rolling(5,min_periods=3).mean()
f=trend*(0.5+0.5*agree)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); x=a.ic
print('assets',len(D),'dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
for hz in [1,3,5]:
 rr=p.pct_change(hz).shift(-hz); vals=[]
 for i in range(len(p)-hz):
  z=pd.concat([f.iloc[i].rename('f'),rr.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append(z.f.corr(z.y))
 print('horizon',hz,'IC',np.nanmean(vals),'n',len(vals))
f.to_csv('scripts/miner_2_20301003_downside_trend_signal.csv')
