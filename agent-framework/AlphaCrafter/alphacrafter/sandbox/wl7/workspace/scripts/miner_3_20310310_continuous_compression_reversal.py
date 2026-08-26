import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>80:
  z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); px[s]=z.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Continuous compression-weighted reversal: fade lagged 5d risk-normalized shock,
# with larger weight for assets in the lower short/long volatility-ratio tail.
rv10=r.rolling(10).std().shift(1); rv40=r.rolling(40).std().shift(1)
comp=(rv10/rv40.replace(0,np.nan)).clip(0.25,4.0)
shock=r.rolling(5).sum().shift(1)/(r.rolling(20).std().shift(1)*np.sqrt(5)).replace(0,np.nan)
w=(1-comp.rank(axis=1,pct=True)).clip(0,1)
sig=(-shock*w).sub((-shock*w).median(axis=1),axis=0)

def test(h):
 y=P.shift(-h)/P-1; a=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   q=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman');a.append(q);rows.append((dt,q,int(v.sum())))
 a=pd.Series(a); return a,rows
for h in [1,5,10,20]:
 a,rows=test(h); print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rows=test(1)
print('rows',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',[a.iloc[i:j].mean() for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20310310_continuous_compression_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310310_continuous_compression_reversal_signal.csv',index=False)
