import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>150:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Candidate: multi-horizon residual reversal, volatility scaled. All inputs are lagged
# one completed session before being used; longer horizons reduce daily noise.
res=[]
for h,w in [(5,.2),(10,.3),(20,.5)]:
 q=P.pct_change(h); res.append(w*(q-q.mean(axis=1).values[:,None]))
raw=-sum(res)
vol=.6*r.ewm(span=20,min_periods=15).std()+.4*r.rolling(60,min_periods=35).std()
sig=(raw/vol).shift(1)
def run(h):
 y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   ic=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'); vals.append(ic); rows.append((dt,ic,int(v.sum())))
 return pd.Series(vals),rows
for h in [1,5,10,20]:
 a,rr=run(h); print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rr]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rr=run(20)
print('assets',len(P.columns),'calendar_coverage %.6f turnover %.6f'%((sig.notna()).mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('segments',[round(a.iloc[i:j].mean(),8) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rr,columns=['date','ic','n']).to_csv('scripts/miner_2_20310922_multihorizon_residual_reversal20_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310922_multihorizon_residual_reversal20_signal.csv',index=False)
