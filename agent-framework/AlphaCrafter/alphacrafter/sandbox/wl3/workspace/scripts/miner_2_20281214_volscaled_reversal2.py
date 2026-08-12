import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None:continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index()
 vol=d.close.pct_change().rolling(20,min_periods=15).std()
 # negative normalized two-session return: short-horizon mean reversion
 f=-(d.close.pct_change(2)/(vol*np.sqrt(2))).replace([np.inf,-np.inf],np.nan)
 r=d.close.shift(-1)/d.close-1
 q=pd.DataFrame({'factor':f,'forward_return_1d':r}).dropna().reset_index();q['symbol']=s;rows.append(q)
q=pd.concat(rows,ignore_index=True);q.to_csv('scripts/miner_2_20281214_volscaled_reversal2_signal.csv',index=False)
ics=[]
for dt,g in q.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.forward_return_1d.nunique()>1:ics.append((dt,g.factor.corr(g.forward_return_1d,method='spearman')))
a=pd.Series(dict(ics)).dropna();print('dates',len(a),'instruments',q.symbol.nunique(),'rows',len(q),'coverage',len(q)/(len(a)*15),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2027-12-31'),('2028','2028-12-31'),('2028-08-01','2028-12-14')]:
 z=a[(a.index>=lo)&(a.index<=hi)];print('regime',lo,hi,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std() if len(z)>1 else np.nan)
q=q.sort_values(['symbol','date']);turn=q.groupby('symbol').factor.apply(lambda x:(x.rank(pct=True).diff().abs()>0.1).mean()).mean();print('turnover_proxy',turn)
print('start',q.date.min(),'end',q.date.max())
