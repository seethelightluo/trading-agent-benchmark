import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,6000).set_index('date')['close'].astype(float).rename(s) for s in U}
P=pd.concat(D.values(),axis=1).sort_index().ffill(); R=P.pct_change()
# Trend-agreement reversal: reverse the recent 10d move, but attenuate reversal
# when the causal 60d trend disagrees with the short-term move. Volatility scale.
short=P.pct_change(10); long=P.pct_change(60); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
agree=np.sign(short)*np.sign(long)
# agreement=+1 means short move follows long trend; retain reversal but 0.5x there;
# disagreement means stronger mean-reversion (1.5x), all causal and interpretable.
mult=(1.0-0.5*agree).clip(0.5,1.5)
f=(-short*mult/(vol+1e-12)).replace([np.inf,-np.inf],np.nan)
fr=P.shift(-10).div(P)-1
rows=[]; ds=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): rows.append(c); ds.append(dt); ns.append(len(z))
ic=pd.Series(rows,index=pd.DatetimeIndex(ds)); print('assets',len(P.columns),'rows',len(P),'start',P.index.min().date(),'end',P.index.max().date())
print('10d dates',len(ic),'meanN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for a,b in [('2023-11-13','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-10-10')]:
 q=ic.loc[a:b]; print('regime',a,b,len(q),round(q.mean(),6) if len(q) else None)
for h in [5,20]:
 q=P.shift(-h).div(P)-1; x=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=np.array(x); print('horizon',h,'dates',len(x),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/np.nanstd(x,ddof=1),6))
out=f.loc['2020':].stack().rename('factor_value').reset_index(); out.columns=['date','symbol','factor_value']; out.to_csv('scripts/miner_2_20351011_trend_agreement_reversal_signal.csv',index=False); print('artifact rows',len(out))
