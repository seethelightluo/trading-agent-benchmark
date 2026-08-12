import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);D[s]=d.drop_duplicates('date').set_index('date')
px=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index()
# Close-to-open/true range pressure: fade prior session directional close displacement.
op=pd.DataFrame({s:d.open.astype(float) for s,d in D.items()}).reindex(px.index)
hi=pd.DataFrame({s:d.high.astype(float) for s,d in D.items()}).reindex(px.index);lo=pd.DataFrame({s:d.low.astype(float) for s,d in D.items()}).reindex(px.index)
# bounded candle pressure, smoothed 3 sessions, lagged
tr=(hi-lo).div(px).replace(0,np.nan)
pressure=((px-op)/op).div(tr).clip(-3,3)
sig=(-pressure.rolling(3,min_periods=3).mean()).shift(1)
def calc(f):
 out=[]
 for t in sig.index:
  z=pd.concat([sig.loc[t].rename('s'),f.loc[t].rename('r')],axis=1).dropna()
  if len(z)>=8:out.append((t,spearmanr(z.s,z.r).statistic,len(z)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
def st(q):return q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()
R=calc(px.shift(-1).div(px)-1)
print('dates',len(px),'instruments',len(D),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(sig.notna().sum(axis=1).mean()/len(U),4))
for lab,q in [('full',R),('2020-22',R.loc['2020':'2022']),('2023-25',R.loc['2023':'2025']),('2026-27',R.loc['2026':'2027']),('2028+',R.loc['2028':]),('recent250',R.tail(250))]:
 if len(q)>2:print(lab,'IC',round(st(q)[0],6),'ICIR',round(st(q)[1],6),'hit',round(st(q)[2],4),'n',len(q))
for h in [3,5,10]:
 q=calc(px.shift(-h).div(px)-1);print('decay',h,'IC',round(st(q)[0],6),'ICIR',round(st(q)[1],6),'n',len(q))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out='scripts/miner_1_20300502_candle_pressure3_signal.csv';sig.to_csv(out,index_label='date');print('artifact',out)
