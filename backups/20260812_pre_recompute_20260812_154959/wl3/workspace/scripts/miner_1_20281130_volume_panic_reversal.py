import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
def fetch(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<200: d=get_index_daily_data(s,4000)
 return d
for s in U:
 d=fetch(s)
 if d is None or len(d)<100: continue
 d=d.copy();d['date']=pd.to_datetime(d.date);d=d.set_index('date').sort_index()
 close=d.close.replace(0,np.nan); ret=close.pct_change(); vol=ret.rolling(20,min_periods=15).std()
 # Volume surprise is deliberately capped; high-volume recent drawdowns get stronger reversal score.
 lv=np.log(d.volume.replace(0,np.nan)); vz=(lv-lv.rolling(60,min_periods=30).mean())/lv.rolling(60,min_periods=30).std()
 f=(-ret.rolling(3,min_periods=3).sum()/vol)*(1+0.5*vz.clip(-1,2))
 # one-session lag: only completed prior session is used at decision
 f=f.shift(1)
 for dt,v in f.items(): rows.append((dt,s,v))
x=pd.DataFrame(rows,columns=['date','symbol','signal']);x=x.dropna()
# aligned forward returns from each instrument
rets=[]
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
 for h in [1,3,5,10]:
  q=d.close.shift(-h)/d.close-1
  rets.append(q.rename(s).rename_axis('date').reset_index().assign(symbol=s,h=h,forward=q.values))
# simple merged long returns
rr=[]
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.drop_duplicates('date').set_index('date').sort_index()
 for h in [1,3,5,10]:
  z=pd.DataFrame({'date':d.index,'symbol':s,'forward':d.close.shift(-h)/d.close-1,'h':h});rr.append(z)
r=pd.concat(rr,ignore_index=True); z=x.merge(r,on=['date','symbol'])
def calc(h):
 a=[]
 for dt,g in z[z.h==h].groupby('date'):
  g=g.dropna(subset=['signal','forward'])
  if len(g)>=8 and g.signal.nunique()>1 and g.forward.nunique()>1:
   a.append(spearmanr(g.signal,g.forward).statistic)
 a=pd.Series(a);return len(a),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(len(a)),(a>0).mean()
print('cutoff',x.date.max().date(),'calendar dates',x.date.nunique(),'instruments',x.symbol.nunique(),'avg valid',x.groupby('date').size().mean(),'coverage',len(x)/(x.date.nunique()*15))
for h in [1,3,5,10]:print('horizon',h,'obs IC ICIR hit',calc(h))
for name,mask in [('2020-22',z.date<pd.Timestamp('2023-01-01')),('2023-25',(z.date>=pd.Timestamp('2023-01-01'))&(z.date<pd.Timestamp('2026-01-01'))),('2026+',z.date>=pd.Timestamp('2026-01-01')),('recent120',z.date>=z.date.max()-pd.Timedelta(days=180))]:
 zz=z[mask];a=[]
 for dt,g in zz[zz.h==1].groupby('date'):
  if len(g)>=8:a.append(spearmanr(g.signal,g.forward).statistic)
 a=pd.Series(a);print('regime',name,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(len(a)) if len(a)>1 else np.nan)
x.to_csv('scripts/miner_1_20281130_volume_panic_reversal_signal.csv',index=False)
