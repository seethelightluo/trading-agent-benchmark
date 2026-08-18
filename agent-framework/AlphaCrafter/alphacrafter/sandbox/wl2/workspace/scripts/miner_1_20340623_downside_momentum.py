import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for fun in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fun(s, days=6000)
            if d is not None and len(d): return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
        except (FileNotFoundError,Exception): pass
    return pd.Series(dtype=float)
px={s:load(s) for s in U}; close=pd.DataFrame(px).sort_index(); print('loaded',close.notna().sum().to_dict())
ret=close.pct_change(); raw=close/close.shift(60)-1; neg=ret.where(ret<0,0).rolling(20,min_periods=15).std(); f=(raw/(neg*np.sqrt(252)+0.02)).replace([np.inf,-np.inf],np.nan)
for h in [10,20,40]:
 fr=close.shift(-h)/close-1; vals=[]
 for dt in close.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); q.to_csv(f'scripts/miner_1_20340623_downside_momentum_{h}d_ic.csv')
 if len(q): print(h,'dates',len(q),'avg_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252),'hit',(q.ic>0).mean())
f.to_csv('scripts/miner_1_20340623_downside_momentum_signal.csv',index_label='date')
print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for start in ['2020-01-01','2026-01-01','2030-01-01','2032-01-01']:
 fr=close.shift(-10)/close-1; q=[]
 for dt in close.index[close.index>=start]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 if q: print('regime',start,'n',len(q),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(252))
