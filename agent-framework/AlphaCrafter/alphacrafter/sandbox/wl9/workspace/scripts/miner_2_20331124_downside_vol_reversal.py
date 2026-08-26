import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change(); r60=cl.pct_change(60)
down=np.sqrt((r.clip(upper=0)**2).rolling(60).mean())*np.sqrt(252)
sig=(-r60/(down+0.04)).clip(-5,5).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for h in [10,20,40,60]:
 f=cl.shift(-h)/cl-1; xs=[]; ns=[]
 for dt in sig.index:
  a,b=sig.loc[dt],f.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: xs.append(a[ok].corr(b[ok],method='spearman'));ns.append(ok.sum())
 x=pd.Series(xs).dropna();print('H',h,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)))
print('coverage %.6f turnover %.6f'%(sig.notna().sum(axis=1).div(15).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
f=cl.shift(-60)/cl-1
for name,m in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033YTD',sig.index.year==2033)]:
 xs=[]
 for dt in sig.index[m]:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:xs.append(sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman'))
 x=pd.Series(xs).dropna();print(name,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()) if len(x)>1 else 'insufficient')
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20331124_downside_vol_reversal_signal.csv',index=False)
