import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}; hi={}; lo={}; vol={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date); z=d.set_index('date')
  cl[s]=z.close; hi[s]=z.high; lo[s]=z.low; vol[s]=z.volume
P=pd.DataFrame(cl).sort_index(); H=pd.DataFrame(hi).reindex(P.index); L=pd.DataFrame(lo).reindex(P.index); V=pd.DataFrame(vol).reindex(P.index)
r=P.pct_change();
# Reversal of recent shock, amplified by an unusually wide range but capped for robustness.
shock=r.rolling(3,min_periods=3).sum().shift(1)
true_range=(H-L)/P
range_z=(true_range-true_range.rolling(20,min_periods=10).median())/(true_range.rolling(20,min_periods=10).std()+1e-12)
amp=(1+range_z.clip(0,2)).shift(1)
sig=(-shock*amp).sub((-shock*amp).median(axis=1),axis=0)
rows=[]; ics=[]
for dt in sig.index:
 v=sig.loc[dt].notna() & P.shift(-1).loc[dt].notna()
 if v.sum()>=8:
  q=sig.loc[dt,v].corr((P.shift(-1)/P-1).loc[dt,v],method='spearman'); rows.append((dt,q,int(v.sum())));ics.append(q)
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
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20310505_range_shock_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310505_range_shock_signal.csv',index=False)
