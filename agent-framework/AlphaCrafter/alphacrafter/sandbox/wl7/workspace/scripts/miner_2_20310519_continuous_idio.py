import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);cs[s]=d.set_index('date').close
P=pd.DataFrame(cs).sort_index(); r=P.pct_change(); common=r.median(axis=1)
# Continuous dispersion-weighted residual reversal. All statistics lagged one day.
beta=r.rolling(60,min_periods=30).cov(common).div(common.rolling(60,min_periods=30).var(),axis=0)
res5=r.rolling(5,min_periods=5).sum()-beta.mul(common.rolling(5,min_periods=5).sum(),axis=0)
vol=r.rolling(20,min_periods=15).std(); z=(-res5/(vol+1e-12)).shift(1)
disp=r.sub(common,axis=0).rolling(5,min_periods=5).std().mean(axis=1)
q=disp.rolling(120,min_periods=60).quantile(.50).shift(1); scale=disp.rolling(120,min_periods=60).median().shift(1)+1e-12
weight=(disp/q).clip(0.5,2.0).div(2.0).clip(0.25,1.0)
sig=z.mul(weight,axis=0)
y=P.shift(-1)/P-1
rows=[]
for dt in sig.index:
 v=sig.loc[dt].notna()&y.loc[dt].notna()
 if v.sum()>=8: rows.append((dt,sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'),int(v.sum())))
a=pd.Series([x[1] for x in rows])
print('rows',len(P),'assets',len(P.columns),'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]))
print('daily IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20]:
 yy=P.shift(-h)/P-1;b=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&yy.loc[dt].notna()
  if v.sum()>=8:b.append(sig.loc[dt,v].corr(yy.loc[dt,v],method='spearman'))
 b=pd.Series(b); print('h',h,'dates',len(b),'IC %.8f ICIR %.8f'%(b.mean(),b.mean()/b.std(ddof=1)))
print('coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',*[round(a.iloc[i:j].mean(),6) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20310519_continuous_idio_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310519_continuous_idio_signal.csv',index=False)
