import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>120:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); r5=P.pct_change(5); r10=P.pct_change(10)
# Candidate: longer-horizon volatility-scaled residual reversal, with robust EWMA-like blend.
# The signal is lagged one completed session to avoid lookahead.
vol=(0.5*r.ewm(span=20,min_periods=15).std()+0.5*r.rolling(40,min_periods=25).std())
raw=-(0.4*(r5-r5.mean(axis=1).values[:,None])+0.6*(r10-r10.mean(axis=1).values[:,None]))
sig=(raw/vol).shift(1)
def run(h):
 y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   ic=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'); vals.append(ic); rows.append((dt,ic,int(v.sum())))
 a=pd.Series(vals)
 return a,rows
for h in [1,5,10,20]:
 a,rr=run(h); print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rr]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rr=run(20)
print('assets',len(P.columns),'calendar_coverage %.6f turnover %.6f'%((sig.notna()).mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('segments',[round(a.iloc[i:j].mean(),8) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rr,columns=['date','ic','n']).to_csv('scripts/miner_2_20310908_ewma40_residual_reversal20_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310908_ewma40_residual_reversal20_signal.csv',index=False)
