import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); cs[s]=d.set_index('date').close
P=pd.DataFrame(cs).sort_index(); r=P.pct_change()
# Idiosyncratic short shock reversal: remove contemporaneous cross-asset market move,
# normalize by each asset's downside volatility, and lag all inputs one day.
market=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(market).div(market.rolling(60,min_periods=40).var(),axis=0)
res=r.sub(beta.mul(market,axis=0),axis=0)
down=r.where(r<0).rolling(30,min_periods=15).std()*np.sqrt(5)
sig=-(res.rolling(5,min_periods=5).sum()/(down+1e-8)).shift(1)
sig=sig.rank(axis=1,pct=True).sub(.5)
y=P.shift(-1)/P-1
vals=[]; ns=[]; dates=[]
for dt in sig.index:
 v=sig.loc[dt].notna()&y.loc[dt].notna()
 if v.sum()>=8:
  vals.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman')); ns.append(v.sum()); dates.append(dt)
a=pd.Series(vals,index=pd.to_datetime(dates))
print('rows',len(P),'assets',len(P.columns),'dates',len(a),'avg_n %.2f'%np.mean(ns))
print('daily IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20]:
 yy=P.shift(-h)/P-1; q=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&yy.loc[dt].notna()
  if v.sum()>=8:q.append(sig.loc[dt,v].corr(yy.loc[dt,v],method='spearman'))
 q=pd.Series(q);print('h',h,'dates',len(q),'IC %.8f ICIR %.8f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage %.5f turnover %.5f'%((sig.notna()).mean().mean(),sig.diff().abs().mean().mean()))
for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]:print('regime',i,j,'IC %.8f'%a.iloc[i:j].mean())
pd.DataFrame({'date':a.index,'ic':a.values}).to_csv('scripts/miner_3_20310602_idio_shock_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310602_idio_shock_reversal_signal.csv',index=False)
