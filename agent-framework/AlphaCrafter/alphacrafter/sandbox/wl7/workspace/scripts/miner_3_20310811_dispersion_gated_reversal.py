import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); ret=P.pct_change(); q=P.pct_change(20)
# Residual reversal, activated only in unusually dispersed cross-asset regimes.
disp=q.std(axis=1); threshold=disp.rolling(60,min_periods=40).median()
base=-q.sub(q.mean(axis=1),axis=0)
sig=base.where(disp.gt(threshold)).shift(1)
Y={h:P.shift(-h)/P-1 for h in [1,5,10]}
allrows=[]
for h,y in Y.items():
 vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   ic=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'); vals.append(ic); rows.append((dt,ic,int(v.sum())))
 a=pd.Series(vals)
 print('h',h,'dates',len(a),'avg_n',round(np.mean([z[2] for z in rows]),2),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 if h==1: allrows=rows
# regime split and signal diagnostics
A=pd.Series([x[1] for x in allrows]); print('daily regimes',[round(A.iloc[i:j].mean(),8) for i,j in [(0,len(A)//3),(len(A)//3,2*len(A)//3),(2*len(A)//3,len(A))]])
print('coverage %.5f active_dates %d turnover %.5f'%((sig.notna()).mean().mean(),sig.notna().any(axis=1).sum(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
pd.DataFrame(allrows,columns=['date','ic','n']).to_csv('scripts/miner_3_20310811_dispersion_gated_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310811_dispersion_gated_reversal_signal.csv',index=False)
