import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# High-dispersion relative shock reversal: reverse each asset's 5d return
# relative to the universe median, only when contemporaneous cross-sectional
# 5d dispersion is above its trailing 120d median; lag signal one session.
r5=r.rolling(5,min_periods=5).sum()
med=r5.median(axis=1)
rel=r5.sub(med,axis=0)
disp=r5.std(axis=1)
state=disp > disp.rolling(120,min_periods=60).median()
vol=r.rolling(20,min_periods=15).std()
sig=(-rel/vol.replace(0,np.nan)).where(state, np.nan).shift(1)
for h in [5,10,20]:
 Y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&Y.loc[dt].notna()
  if v.sum()>=8:
   c=sig.loc[dt,v].corr(Y.loc[dt,v],method='spearman')
   if pd.notna(c): vals.append(c); rows.append((dt,c,int(v.sum())))
 a=pd.Series(vals); n=len(a)
 print('H',h,'dates',n,'avg_n %.2f'%np.mean([x[2] for x in rows]) if rows else 0)
 print('IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 print('segments',[round(a.iloc[i:j].mean(),8) for i,j in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
 if h==20:
  sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20320308_highdisp_shock_signal.csv',index=False)
  pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20320308_highdisp_shock_ic.csv',index=False)
print('assets',len(P.columns),'coverage %.6f turnover %.6f state_rate %.6f'%((sig.notna()).mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean(),state.mean()))
