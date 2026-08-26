import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>80:
  z=d[['date','open','close']].copy(); z.date=pd.to_datetime(z.date); raw[s]=z.set_index('date')
# completed-day OHLC. Fade overnight gap, normalized by trailing gap volatility and centered cross-section.
gap=pd.DataFrame({s:(z.open/z.close.shift(1)-1) for s,z in raw.items()}).sort_index()
vol=gap.rolling(20,min_periods=10).std().shift(1)
sig=(-gap/vol.replace(0,np.nan)).shift(1)
sig=sig.sub(sig.median(axis=1),axis=0)
P=pd.DataFrame({s:z.close for s,z in raw.items()}).sort_index()
def test(h):
 y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   q=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'); vals.append(q); rows.append((dt,q,int(v.sum())))
 a=pd.Series(vals)
 return a,rows
for h in [1,5,10,20]:
 a,rows=test(h); print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rows=test(1)
print('rows',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',[a.iloc[i:j].mean() for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20310324_overnight_gap_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310324_overnight_gap_reversal_signal.csv',index=False)
print('signal_artifact=scripts/miner_3_20310324_overnight_gap_reversal_signal.csv')
