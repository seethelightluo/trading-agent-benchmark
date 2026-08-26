import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); ret20=P.pct_change(20)
vol=r.rolling(40,min_periods=30).std()
def z(x): return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1).replace(0,np.nan),axis=0)
# Unequal blend: emphasize the more stable residual reversal while retaining acceleration.
rev=-ret20.sub(ret20.mean(axis=1),axis=0)
acc=(P.pct_change(10)-P.pct_change(20).shift(10))/vol
sig=(0.65*z(rev)+0.35*z(acc)).shift(1)
def evaluate(h):
 y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   vals.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman')); rows.append((dt,vals[-1],int(v.sum())))
 a=pd.Series(vals)
 return a,rows
for h in [1,5,10]:
 a,rows=evaluate(h)
 print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rows=evaluate(1)
print('history_dates',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%((sig.notna()).mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',[round(a.iloc[i:j].mean(),8) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20310825_unequal_blend_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310825_unequal_blend_signal.csv',index=False)
