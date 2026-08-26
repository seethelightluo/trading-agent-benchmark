import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,6000)
 if d is None or len(d)<100: d=get_index_daily_data(s,6000)
 raw[s]=d
prices=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill()
ret=prices.pct_change()
# lagged 60-session return divided by downside deviation; all inputs through t-1
mom=prices.shift(1).pct_change(60)
down=ret.where(ret<0).rolling(60,min_periods=40).std().shift(1)
factor=mom/down.replace(0,np.nan)
# forward non-overlapping 10 sessions from t close to t+10 close
fwd=prices.shift(-10)/prices-1
rows=[]; sigrows=[]
for dt in factor.index:
    x=factor.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)<8: continue
    ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if np.isfinite(ic):
      rows.append((dt,ic,len(z)))
      for s in z.index: sigrows.append((dt,s,float(x[s])))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# rank turnover among consecutive valid dates
sig=pd.DataFrame(sigrows,columns=['date','symbol','signal'])
piv=sig.pivot(index='date',columns='symbol',values='signal').rank(axis=1,pct=True)
to=piv.diff().abs().mean(axis=1).mean()
mean=ic.ic.mean(); sd=ic.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
print('dates',len(ic),'avg_n',ic.n.mean(),'coverage',len(sig)/(len(factor)*len(U)))
print('ic10',mean,'daily_paper_icir',icir,'hit', (ic.ic>0).mean(),'turnover',to)
for h in [1,5,20]:
 fy=prices.shift(-h)/prices-1; vals=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(vals),len(vals))
ic.to_csv('scripts/miner_1_20350802_downside_efficiency_momentum_ic.csv')
sig.to_csv('scripts/miner_1_20350802_downside_efficiency_momentum_signal.csv',index=False)
