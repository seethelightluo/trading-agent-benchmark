import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>160:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); V=R.rolling(60,min_periods=45).std()
def z(x): return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1).replace(0,np.nan),axis=0)
# Slow-volatility normalization makes the signal less sensitive to transient volatility spikes.
rev=-P.pct_change(30)/V
acc=(P.pct_change(15)-P.pct_change(15).shift(15))/V
sig=(.70*z(rev)+.30*z(acc)).shift(1)
def ev(h):
 y=P.shift(-h)/P-1; o=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:o.append((dt,sig.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'),int(ok.sum())))
 a=pd.Series([x[1] for x in o]); return a,o
for h in [1,5,10]:
 a,o=ev(h); print('h',h,'dates',len(a),'avg_n',round(np.mean([x[2] for x in o]),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),5))
a,o=ev(1); print('history_dates',len(P),'assets',len(P.columns),'coverage',round(sig.notna().mean().mean(),6),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for k,(i,j) in enumerate([(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]): print('regime',k+1,round(a.iloc[i:j].mean(),8),'dates',j-i)
pd.DataFrame(o,columns=['date','ic','n']).to_csv('scripts/miner_3_20311103_slowvol_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20311103_slowvol_reversal_signal.csv',index=False)
