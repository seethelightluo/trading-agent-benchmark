import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<180: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
px=pd.concat(D,axis=1).sort_index().ffill(); r=px.pct_change()
csdisp=r.rolling(5,min_periods=3).apply(lambda x: np.std(x),raw=True).mean(axis=1).shift(1); threshold=csdisp.rolling(60,min_periods=30).median()
gate=(csdisp>threshold)
raw=-(px.pct_change(5).shift(1))/r.rolling(20,min_periods=15).std().shift(1)
f=raw.where(gate,0).replace([np.inf,-np.inf],np.nan)
f.to_csv('scripts/miner_2_20350719_dispersion_gated_reversal_signal.csv',index_label='date')
ics={h:[] for h in [5,10,20,40]}; nins=[]; dates=[]
for dt in px.index:
 x=f.loc[dt]
 if x.notna().sum()>=8:
  dates.append(dt); nins.append(x.notna().sum())
  for h in ics:
   yy=px.pct_change(h).shift(-h).loc[dt]; v=x.notna()&yy.notna()
   if v.sum()>=8: ics[h].append(x[v].corr(yy[v],method='spearman'))
print('dates',len(dates),'avg_instruments',np.mean(nins),'universe',len(U),'cell_coverage',f.notna().mean().mean(),'gate_rate',gate.mean())
for h,a in ics.items():
 a=np.array(a); print(h,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(len(a)),'hit',np.mean(a>0),'n',len(a))
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean())
