import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); disp=r.rolling(5,min_periods=4).std().mean(axis=1)
# Short-horizon reversal, only after a broad cross-asset shock; all information lagged.
shock=disp.gt(disp.rolling(60,min_periods=40).quantile(.65))
base=-r.rolling(3,min_periods=3).sum(); sig=base.where(shock).shift(1)
Y=P.shift(-1)/P-1; vals=[]; rows=[]
for dt in sig.index:
 v=sig.loc[dt].notna()&Y.loc[dt].notna()
 if v.sum()>=8:
  ic=sig.loc[dt,v].corr(Y.loc[dt,v],method='spearman'); vals.append(ic);rows.append((dt,ic,int(v.sum())))
a=pd.Series(vals); print('dates',len(a),'avg_n',np.mean([z[2] for z in rows]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10]:
 y=P.shift(-h)/P-1; z=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:z.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'))
 z=pd.Series(z);print('h',h,'dates',len(z),'IC %.8f ICIR %.8f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('coverage %.5f active_dates %d turnover %.5f'%((sig.notna()).mean().mean(),sig.notna().any(axis=1).sum(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',[round(a.iloc[i:j].mean(),8) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20310811_shock_reversal3_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310811_shock_reversal3_signal.csv',index=False)
