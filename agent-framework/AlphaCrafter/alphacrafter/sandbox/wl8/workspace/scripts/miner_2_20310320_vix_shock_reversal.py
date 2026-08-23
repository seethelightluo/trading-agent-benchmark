import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s, days=3000) for s in U}; v=get_index_daily_data('VIX', days=3000)
series={s:df.set_index('date')['close'].astype(float) for s,df in px.items() if df is not None and len(df)>0}
vix=v.set_index('date')['close'].astype(float)
close=pd.DataFrame(series).sort_index(); r5=close.pct_change(5)
vixshock=(vix.pct_change(5)-vix.pct_change(5).rolling(252,min_periods=60).median()).clip(lower=0)
sig=(-r5).mul(vixshock.reindex(close.index).fillna(0).clip(upper=1),axis=0); fwd=close.shift(-10)/close-1
ics=[]; turnovers=[]; counts=[]; prev=None
for d in close.index:
 x=sig.loc[d]; y=fwd.loc[d]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8 and x.loc[z.index].abs().sum()>0:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(ic): ics.append((d,ic)); counts.append(len(z))
 rank=x.rank(pct=True)
 if prev is not None:
  q=pd.concat([rank,prev],axis=1).dropna()
  if len(q): turnovers.append((rank.name,(q.iloc[:,0]-q.iloc[:,1]).abs().mean()))
 prev=rank
iv=np.array([x[1] for x in ics]); dates=[x[0] for x in ics]
def summ(mask):
 a=iv[mask]; sd=np.std(a,ddof=1) if len(a)>1 else 0
 return (float(np.mean(a)),float(np.mean(a)/sd) if sd>0 else 0.,float(np.mean(a>0)),len(a))
print('cutoff',close.index.max(),'dates',len(iv),'avg_inst',np.mean(counts),'coverage',np.mean(counts)/15,'turnover',np.mean([x[1] for x in turnovers]))
print('10d',summ(np.ones(len(iv),bool)))
for n in [180,360]: print('recent',n,summ(np.array(dates)>=dates[-1]-pd.tseries.offsets.BDay(n)))
for yr in sorted(set(d.year for d in dates)): print('year',yr,summ(np.array([d.year==yr for d in dates])))
print('decay')
for h in [5,10,20]:
 ff=close.shift(-h)/close-1; aa=[]
 for d in close.index:
  z=pd.concat([sig.loc[d],ff.loc[d]],axis=1).dropna()
  if len(z)>=8 and sig.loc[d].abs().sum()>0: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(h,float(np.nanmean(aa)),float(np.nanmean(aa)/np.nanstd(aa,ddof=1)),len(aa))
sig.index.name='date'; sig.to_csv('scripts/miner_2_20310320_vix_shock_reversal_signal.csv')
