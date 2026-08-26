import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
r5=r.rolling(5,min_periods=5).sum(); med=r5.median(axis=1); rel=r5.sub(med,axis=0)
disp=r5.std(axis=1); state=disp>disp.rolling(120,min_periods=60).median()
vol=r.rolling(20,min_periods=15).std()
# downside-only relative shocks: only laggards receive reversal score
sig=(-rel.clip(upper=0)/vol.replace(0,np.nan)).where(state,np.nan).shift(1)
for h in [5,10,20]:
 Y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&Y.loc[dt].notna()
  if v.sum()>=8:
   c=sig.loc[dt,v].corr(Y.loc[dt,v],method='spearman')
   if pd.notna(c): vals.append(c); rows.append((dt,c,int(v.sum())))
 a=pd.Series(vals); n=len(a)
 print('H',h,'dates',n,'avg_n %.2f'%(np.mean([x[2] for x in rows]) if rows else 0))
 print('IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 print('segments',[round(a.iloc[i:j].mean(),8) for i,j in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
# provenance signal artifact
sig.to_csv('scripts/miner_2_20320322_downside_highdisp_signal.csv')
