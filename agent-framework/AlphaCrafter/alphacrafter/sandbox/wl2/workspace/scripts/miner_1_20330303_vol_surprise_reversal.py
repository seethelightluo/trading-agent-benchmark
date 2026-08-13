import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
px={s:d.set_index('date').close.astype(float) for s,d in D.items() if d is not None and len(d)>300}
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rv=R.rolling(20,min_periods=15).std(); lv=R.rolling(120,min_periods=80).std()
shock=R.rolling(5,min_periods=4).sum()/(rv*np.sqrt(5)+1e-12)
f=-(shock/(lv/(rv+1e-12)+1e-12)).replace([np.inf,-np.inf],np.nan)
ics={h:[] for h in [1,3,5,10]}; cov=[]
for i in range(len(P)-10):
 x=f.iloc[i]; valid=x.notna(); cov.append(valid.sum()/15)
 if valid.sum()<8: continue
 for h in ics:
  y=P.iloc[i+h]/P.iloc[i]-1; z=valid&y.notna()
  if z.sum()>=8: ics[h].append(x[z].corr(y[z]))
print('dates',len(P),'assets',len(px),'mean_valid',np.mean(cov))
for h,a0 in ics.items():
 a=np.array(a0); print(h,'n',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),'hit',np.mean(a>0),'earlylate',np.nanmean(a[:len(a)//2]),np.nanmean(a[len(a)//2:]))
rr=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rr)):
 z=rr.iloc[i].notna()&rr.iloc[i-1].notna()
 if z.sum()>=8: turn.append(np.mean(np.abs(rr.iloc[i][z]-rr.iloc[i-1][z])))
print('turnover_rank_abs',np.mean(turn),'coverage',np.mean(f.notna().sum(axis=1)/15))
out=f.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20330303_vol_surprise_reversal_signal.csv')
