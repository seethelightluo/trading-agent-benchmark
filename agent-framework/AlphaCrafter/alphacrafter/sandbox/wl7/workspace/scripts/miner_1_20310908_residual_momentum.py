import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Residual momentum: asset 20d return minus contemporaneous equal-weight cross-asset return,
# normalized by asset 30d volatility; lag one day before forward returns.
market=r.mean(axis=1)
resid=r.sub(market,axis=0)
raw=resid.rolling(20,min_periods=20).sum()/(r.rolling(30,min_periods=25).std()*np.sqrt(20)+1e-12)
sig=raw.rank(axis=1,pct=True).shift(1)
print('rows',len(P),'assets',len(P.columns),'start',P.index.min(),'end',P.index.max())
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; vals=[]; ns=[]; ds=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   vals.append(sig.loc[dt,ok].corr(y.loc[dt,ok],method='spearman')); ns.append(int(ok.sum())); ds.append(dt)
 a=pd.Series(vals,index=pd.to_datetime(ds)); print('h',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
 if h==10: pd.DataFrame({'date':a.index,'ic':a.values,'n':ns}).to_csv('scripts/miner_1_20310908_residual_momentum_ic_10d.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310908_residual_momentum_signal.csv',index=False)
print('coverage',round(sig.notna().mean().mean(),6),'turnover',round(sig.diff().abs().mean().mean(),6))
# regime thirds by market trailing 60d return
ic10=pd.read_csv('scripts/miner_1_20310908_residual_momentum_ic_10d.csv',parse_dates=['date']).set_index('date').ic
reg=market.rolling(60,min_periods=50).sum().reindex(ic10.index)
for q in range(3):
 lo,hi=reg.quantile(q/3),reg.quantile((q+1)/3); z=ic10[(reg>=lo)&(reg<=hi)]
 print('regime',q+1,'dates',len(z),'IC',round(z.mean(),8),'ICIR',round(z.mean()/z.std(ddof=1),8))
