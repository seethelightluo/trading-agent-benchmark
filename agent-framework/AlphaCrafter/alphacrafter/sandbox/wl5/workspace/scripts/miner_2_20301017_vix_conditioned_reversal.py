import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,4000)
 return pd.Series(dtype=float) if d is None or len(d)==0 else d.set_index('date')['close'].astype(float).sort_index()
px=pd.DataFrame({s:load(s) for s in ASSETS}).loc[:pd.Timestamp('2030-10-16')]
r=px.pct_change(); vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:pd.Timestamp('2030-10-16')]
vp=v.rolling(252,min_periods=100).rank(pct=True).reindex(px.index).ffill()
stress=0.5+vp.clip(0.0,1.0)
sig=-(px/px.shift(20)-1)/vol*stress.values[:,None]; sig=pd.DataFrame(sig,index=px.index,columns=px.columns)
sig.to_csv('scripts/miner_2_20301017_vix_conditioned_reversal_signal.csv',index_label='date')
fwd=px.shift(-10)/px-1; obs=[]; turns=[]; prev=None
for dt in sig.index:
 x=sig.loc[dt].replace([np.inf,-np.inf],np.nan).dropna(); y=fwd.loc[dt].reindex(x.index).dropna(); x=x.reindex(y.index)
 if len(x)<8: continue
 obs.append((dt,x.corr(y,method='spearman'),len(x))); q=x.rank(pct=True)
 if prev is not None: turns.append((q-prev.reindex(q.index)).abs().mean())
 prev=q
z=pd.DataFrame(obs,columns=['date','ic','n']).dropna(); m=z.ic.mean(); sd=z.ic.std(ddof=1)
print('factor vix_conditioned_reversal_20d'); print('dates',len(z),'mean_n',z.n.mean(),'coverage',z.n.mean()/15,'IC',m,'daily_ICIR',m/sd,'annualized_reference_ICIR',m/sd*np.sqrt(252),'hit',(z.ic>0).mean(),'turnover',np.mean(turns))
for name,a,b in [('2020-24','2020-01-01','2024-12-31'),('2025-27','2025-01-01','2027-12-31'),('2028-29','2028-01-01','2029-12-31'),('2030','2030-01-01','2030-10-16')]:
 q=z[(z.date>=a)&(z.date<=b)]; print(name,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
