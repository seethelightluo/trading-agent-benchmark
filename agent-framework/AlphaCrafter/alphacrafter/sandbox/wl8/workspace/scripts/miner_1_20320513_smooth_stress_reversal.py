import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in watch:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): prices[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(prices).sort_index().ffill(); r=px.pct_change()
disp=r.std(axis=1,ddof=0); med=disp.rolling(60,min_periods=40).median().shift(1)
# Smooth stress intensity, lagged and capped, avoids binary threshold discontinuity
stress=(disp.shift(1)/med).clip(0.5,2.0)
vol=r.rolling(20,min_periods=15).std().shift(1)*np.sqrt(20)
f=(-px.pct_change(5).shift(1)/vol).mul(stress,axis=0)
fwd=px.shift(-10)/px-1
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  z=a[ok].corr(b[ok],method='spearman')
  if pd.notna(z): rows.append((dt,z,ok.sum()))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mean=ic.ic.mean(); sd=ic.ic.std(ddof=1)
print('dates',len(ic),'start',ic.index.min(),'end',ic.index.max(),'avg_n',ic.n.mean())
print('coverage',float(f.notna().mean().mean()),'mean_ic',mean,'icir',mean/sd*np.sqrt(252),'hit',float((ic.ic>0).mean()),'turnover',float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; vals=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8:
   z=f.loc[dt,ok].corr(fw.loc[dt,ok],method='spearman')
   if pd.notna(z): vals.append(z)
 print('decay',h,float(np.mean(vals)),len(vals))
for label,sub in [('365d',ic.tail(252)),('180d',ic.tail(126)),('2032YTD',ic[ic.index>='2032-01-01'])]:
 if len(sub)>5: print(label,'n',len(sub),'ic',sub.ic.mean(),'icir',sub.ic.mean()/sub.ic.std(ddof=1)*np.sqrt(252))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20320513_smooth_stress_reversal_signal.csv',index=False)
ic.to_csv('scripts/miner_1_20320513_smooth_stress_reversal_ic.csv')
