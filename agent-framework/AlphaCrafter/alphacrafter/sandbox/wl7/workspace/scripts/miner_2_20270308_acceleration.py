import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-08')
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d[d.index<=cut]
p=pd.concat(px,axis=1).sort_index().ffill()
# Acceleration: recent 5d return versus average of prior 20d return, volatility normalized; signal is known at t and predicts t+1
ret=p.pct_change()
short=p/p.shift(5)-1
prior=p.shift(5)/p.shift(25)-1
vol=ret.rolling(20).std()
factor=(short-prior/4)/vol
# strict lag so signal at t-1 predicts t return
sig=factor.shift(1); fwd=p.pct_change().shift(-0) # return on date t; signal prior date
rows=[]; vals=[]
for dt in p.index:
 x=sig.loc[dt]; y=ret.loc[dt]
 ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],y[ok]).statistic
  rows.append((dt,ic,ok.sum()))
ics=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
ics=ics[ics.index<=cut]
mean=ics.ic.mean(); std=ics.ic.std(ddof=1); icir=mean/std*np.sqrt(252) if std else np.nan
# rank turnover
r=sig.rank(axis=1,pct=True); to=(r.diff().abs().mean(axis=1)).dropna().mean()
print('dates',len(ics),'avg_n',ics.n.mean(),'coverage',sig.notna().mean().mean(),'IC',mean,'ICIR',icir,'hit',(ics.ic>0).mean(),'turnover',to)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-08')]:
 z=ics.loc[lo:hi].ic; print(lo,hi,len(z),z.mean(),z.mean()/z.std()*np.sqrt(252) if len(z)>1 else np.nan)
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1
 rr=[]
 for dt in p.index:
  x=sig.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: rr.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.nanmean(rr),len(rr))
out=pd.DataFrame(sig.stack(),columns=['signal']); out.index.names=['date','symbol']; out.to_csv('scripts/miner_2_20270308_acceleration_signal.csv')
