import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Candidate: lagged medium-horizon reversal scaled by recent downside risk.
# High values identify assets with poor recent return per unit downside volatility.
down=r.where(r<0,0.0).rolling(40,min_periods=25).std()
sig=(-(P.pct_change(15)/down.replace(0,np.nan))).shift(1)
y={h:P.shift(-h)/P-1 for h in [5,10,20]}
for h,Y in y.items():
 vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&Y.loc[dt].notna()
  if v.sum()>=8:
   ic=sig.loc[dt,v].corr(Y.loc[dt,v],method='spearman'); vals.append(ic); rows.append((dt,ic,int(v.sum())))
 a=pd.Series(vals)
 print('H',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]))
 print('IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 print('segments',[round(a.iloc[i:j].mean(),8) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
 if h==10:
  sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20320126_downside_risk_reversal_signal.csv',index=False)
  pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20320126_downside_risk_reversal_ic.csv',index=False)
print('assets',len(P.columns),'coverage %.6f turnover %.6f'%((sig.notna()).mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
