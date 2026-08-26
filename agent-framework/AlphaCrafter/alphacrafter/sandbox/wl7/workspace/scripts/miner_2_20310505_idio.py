import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);cs[s]=d.set_index('date').close
P=pd.DataFrame(cs).sort_index();r=P.pct_change(); cross=r.median(axis=1)
# Idiosyncratic shock: asset 5d return less its rolling beta to the common cross-asset move; reverse only when dispersion is elevated.
rollcov=r.rolling(60,min_periods=30).cov(cross); rollvar=cross.rolling(60,min_periods=30).var(); beta=rollcov.div(rollvar,axis=0)
idio=(r.rolling(5,min_periods=5).sum()-beta*cross.rolling(5,min_periods=5).sum().values[:,None]).shift(1)
disp=r.sub(cross,axis=0).rolling(5,min_periods=5).std().mean(axis=1)
threshold=disp.rolling(120,min_periods=60).quantile(.65).shift(1)
gate=(disp>threshold).astype(float).replace(0,np.nan)
sig=(-idio/(r.rolling(20,min_periods=15).std().shift(1)+1e-12)*gate).sub((-idio/(r.rolling(20,min_periods=15).std().shift(1)+1e-12)*gate).median(axis=1),axis=0)
rows=[];ics=[]
for dt in sig.index:
 y=P.shift(-1)/P-1;v=sig.loc[dt].notna()&y.loc[dt].notna()
 if v.sum()>=8: rows.append((dt,sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'),int(v.sum())));ics.append(rows[-1][1])
a=pd.Series(ics)
print('rows',len(P),'assets',len(P.columns),'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]))
print('daily IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20]:
 y=P.shift(-h)/P-1;b=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:b.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'))
 b=pd.Series(b);print('h',h,'dates',len(b),'IC %.8f ICIR %.8f'%(b.mean(),b.mean()/b.std(ddof=1)))
print('coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',*[round(a.iloc[i:j].mean(),6) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20310505_idio_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310505_idio_signal.csv',index=False)
