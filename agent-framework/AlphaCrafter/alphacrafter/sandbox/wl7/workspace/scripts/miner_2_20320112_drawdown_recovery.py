import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Drawdown-recovery reversal: trailing return relative to worst peak-to-trough loss,
# lagged one completed session; high values favor assets with poor recovery efficiency.
peak=P.rolling(20,min_periods=15).max()
dd=(P/peak-1).rolling(20,min_periods=15).min().abs()
sig=(-(P.pct_change(20)/dd.replace(0,np.nan))).shift(1)
y=P.shift(-10)/P-1
vals=[]; rows=[]
for dt in sig.index:
 v=sig.loc[dt].notna()&y.loc[dt].notna()
 if v.sum()>=8:
  ic=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'); vals.append(ic); rows.append((dt,ic,int(v.sum())))
a=pd.Series(vals)
print('assets',len(P.columns),'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]))
print('IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('coverage %.6f turnover %.6f'%((sig.notna()).mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('segments',[round(a.iloc[i:j].mean(),8) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20320112_drawdown_recovery_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20320112_drawdown_recovery_signal.csv',index=False)
