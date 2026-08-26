import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>120:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Trend consistency: medium-term return risk-scaled by volatility and persistence of
# positive sessions. All inputs are lagged one completed session.
vol=r.rolling(60,min_periods=40).std()
persistence=(r>0).rolling(40,min_periods=30).mean()
sig=((P.pct_change(60)/vol)*persistence).shift(1)
for h in [5,10,20]:
 Y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&Y.loc[dt].notna()
  if v.sum()>=8:
   vals.append(sig.loc[dt,v].corr(Y.loc[dt,v],method='spearman')); rows.append((dt,vals[-1],int(v.sum())))
 a=pd.Series(vals); print('H',h,'dates',len(a),'avg_n %.2f'%np.mean([z[2] for z in rows]))
 print('IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 n=len(a); print('segments',[round(a.iloc[i:j].mean(),8) for i,j in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
 if h==10:
  sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20320209_trend_consistency_signal.csv',index=False)
  pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20320209_trend_consistency_ic.csv',index=False)
print('assets',len(P.columns),'coverage %.6f turnover %.6f'%((sig.notna()).mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
