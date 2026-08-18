import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[])
if not U: U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<120: d=get_index_daily_data(s,5000)
 if d is not None and len(d)>120:
  z=d[['date','close']].copy(); z['date']=pd.to_datetime(z.date); px[s]=z.set_index('date').close
P=pd.DataFrame(px).sort_index().ffill()
R=P.pct_change()
# Trend-persistence: medium horizon return, weighted by fraction of positive days;
# lag one completed day to avoid look-ahead.
ret20=P.pct_change(20); pos20=R.rolling(20).mean(); vol20=R.rolling(20).std()
F=(ret20*pos20).shift(1)
# forward 10d returns from each date
FR=P.shift(-10)/P-1
rows=[]
for dt in F.index:
 x=F.loc[dt]; y=FR.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ic=x[ok].corr(y[ok]); rows.append((dt,ic,ok.sum()))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('universe',len(U),'usable',len(px),'dates',len(q),'avg_n',q.n.mean())
print('IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean()))
for a,b in [('2026-07-16','2029-12-31'),('2030-01-01','2033-12-31'),('2029-01-01','2033-12-31'),('2033-01-01','2034-02-03')]:
 z=q.loc[a:b].ic.dropna(); print(a,b,'n',len(z),'IC %.8f ICIR %.8f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1), (z>0).mean()))
# daily signal turnover, cross-sectional rank changes
rank=F.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('coverage',F.notna().sum().sum()/(F.shape[0]*F.shape[1]),'turnover',turn)
q.to_csv('scripts/miner_1_20340203_trend_persistence_20d_ic.csv')
F.to_csv('scripts/miner_1_20340203_trend_persistence_20d_signal.csv')
