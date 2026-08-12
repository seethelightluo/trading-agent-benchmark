import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(s):
    d=get_stock_daily_data(s,days=4000)
    if d is None: d=get_index_daily_data(s,days=4000)
    return d
frames={}
for s in U:
 d=get(s)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'); d=d.set_index('date')
  frames[s]=d
close=pd.DataFrame({s:d.close for s,d in frames.items()}).sort_index().astype(float)
r=close.pct_change()
# path-efficiency trend: signed 20d return divided by sum absolute daily returns, with volatility moderation
ret20=close/close.shift(20)-1
path=r.rolling(20).apply(lambda x: np.abs(x).sum(),raw=True)
vol=r.rolling(20).std()
f=ret20/(path+1e-8) / (vol+1e-8)**0.25
# winsorize cross section not needed; save signals
rows=[]; ics={h:[] for h in [1,5,10]}; counts=[]; turnover=[]
for i,date in enumerate(close.index):
 if i+10>=len(close): break
 x=f.loc[date].replace([np.inf,-np.inf],np.nan)
 valid=x.notna() & close.loc[date].notna()
 n=int(valid.sum())
 if n<8: continue
 counts.append(n)
 for h in ics:
  y=close.shift(-h).loc[date]/close.loc[date]-1
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: ics[h].append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 if len(rows)==0 or i%10==0:
  turnover.append(np.nan)
 else:
  prev=f.iloc[i-10].replace([np.inf,-np.inf],np.nan)
  a=x.notna()&prev.notna()
  turnover.append((x[a].rank().sub(prev[a].rank()).abs().mean()/max(1,n)))
 for s,v in x.items(): rows.append({'date':date,'symbol':s,'signal':v})
print('assets',len(frames),'dates',len(ics[1]),'avg_n',np.mean(counts),'coverage',np.mean(counts)/15)
for h,a in ics.items():
 a=np.array(a); print('H',h,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'n',len(a))
print('turnover_proxy',np.nanmean(turnover))
for start in ['2025-01-01','2028-01-01','2029-01-01','2029-07-01']:
 a=[]
 for date,v in zip(close.index, range(len(close.index))):
  if date>=start and v<len(close)-10:
   x=f.loc[date].replace([np.inf,-np.inf],np.nan); y=close.shift(-1).loc[date]/close.loc[date]-1
   z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a); print(start,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1) if len(a)>1 else np.nan)
out=pd.DataFrame(rows); out.to_csv('scripts/miner_3_20300418_path_efficiency_signal.csv',index=False)
