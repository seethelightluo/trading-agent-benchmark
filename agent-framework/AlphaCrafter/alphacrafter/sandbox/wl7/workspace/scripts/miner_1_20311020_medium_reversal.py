import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>120:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Medium-horizon reversal: 10-day return normalized by 40-day realized volatility, lagged one day.
vol=r.rolling(40,min_periods=25).std()
sig=(-P.pct_change(10).div(vol)).shift(1)
def ev(h):
 y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   ic=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'); vals.append(ic); rows.append((dt,ic,int(v.sum())))
 a=pd.Series(vals); return a,rows
for h in [1,5,10,20]:
 a,rows=ev(h); print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([z[2] for z in rows]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rows=ev(10); n=len(a)
print('history_dates',len(P),'assets',len(P.columns),'coverage %.5f'%sig.notna().mean().mean())
print('third_IC',[round(a.iloc[i:j].mean(),8) for i,j in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
# rank turnover, measured only where both dates have valid ranks
q=sig.rank(axis=1,pct=True); print('turnover %.5f'%q.diff().abs().mean().mean())
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20311020_medium_reversal_ic_10d.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20311020_medium_reversal_signal.csv',index=False)
