import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
r5=r.rolling(5,min_periods=4).sum(); vol20=r.rolling(20,min_periods=15).std()*np.sqrt(20)
# Cross-asset dispersion is computed cross-sectionally, then smoothed, and lagged.
xs_disp=r.std(axis=1).rolling(20,min_periods=15).mean()
threshold=xs_disp.rolling(60,min_periods=40).quantile(.60)
gate=(xs_disp>threshold).astype(float)
sig=(-r5/(vol20+1e-12)).mul(gate,axis=0).shift(1)
sig=sig.rank(axis=1,pct=True).sub(.5)
def test(h):
 y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   z=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'); vals.append(z); rows.append((dt,z,int(v.sum())))
 a=pd.Series(vals); return a,rows
for h in [1,5,10,20]:
 a,rows=test(h); print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rows=test(1)
print('rows',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%((sig.notna()).mean().mean(),sig.diff().abs().mean().mean()))
print('regimes',[round(a.iloc[i:j].mean(),8) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20310630_dispersion_weighted_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310630_dispersion_weighted_reversal_signal.csv',index=False)
