import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100: d.date=pd.to_datetime(d.date); cs[s]=d.set_index('date').close
P=pd.DataFrame(cs).sort_index(); r=P.pct_change()
# Dispersion-blended shock reversal: combine short and medium shocks, normalized by each asset's realized risk.
vol=r.rolling(30,min_periods=20).std().shift(1)
sig=-(.7*r.rolling(3,min_periods=3).sum()+.3*r.rolling(10,min_periods=10).sum())/(vol*np.sqrt(5)+1e-12)
sig=sig.shift(1)
y=P.shift(-1)/P-1
all_a=[]; ns=[]
for dt in sig.index:
 v=sig.loc[dt].notna()&y.loc[dt].notna()
 if v.sum()>=8: all_a.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'));ns.append(v.sum())
a=pd.Series(all_a); print('rows',len(P),'assets',len(P.columns),'dates',len(a),'avg_n %.2f'%np.mean(ns));print('daily IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20]:
 z=P.shift(-h)/P-1;b=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&z.loc[dt].notna()
  if v.sum()>=8:b.append(sig.loc[dt,v].corr(z.loc[dt,v],method='spearman'))
 b=pd.Series(b);print('h',h,'dates',len(b),'IC %.8f ICIR %.8f'%(b.mean(),b.mean()/b.std(ddof=1)))
print('coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()));print('regimes',*[round(a.iloc[i:j].mean(),6) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame({'ic':a}).to_csv('scripts/miner_3_20310505_shock_reversal_ic.csv',index=False);sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310505_shock_reversal_signal.csv',index=False)
