import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
frames={}
for f in files:
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
 frames[s]=d['close'].astype(float)
p=pd.DataFrame(frames).sort_index(); p=p.loc[:'2034-07-19']
# lagged signal: 20d momentum divided by 20d realized vol, with 5d smoothing
r=p.pct_change(); mom=p.shift(1).pct_change(20); vol=r.shift(1).rolling(20).std()*np.sqrt(20)
f=(mom/vol).rolling(5).mean()
ics=[]; rows=[]
for dt in p.index:
 if dt not in f.index: continue
 fut=p.shift(-10).loc[dt]/p.loc[dt]-1
 x=f.loc[dt]; ok=x.notna()&fut.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],fut[ok]).statistic
  ics.append(ic); rows.append({'date':dt,'ic':ic,'n':int(ok.sum())})
ics=np.array(ics); print('dates',len(ics),'avgN',np.mean([z['n'] for z in rows]),'coverage',f.notna().stack().mean())
print('IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0))
for a,b in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-07-19'),('2033','2034-07-19')]:
 q=[z['ic'] for z in rows if pd.Timestamp(a)<=z['date']<=pd.Timestamp(b)]
 print(a,b,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
# decay horizons
for h in [1,5,10,20]:
 vals=[]
 for dt in p.index:
  if dt not in f.index: continue
  fut=p.shift(-h).loc[dt]/p.loc[dt]-1; x=f.loc[dt]; ok=x.notna()&fut.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],fut[ok]).statistic)
 print('decay',h,np.mean(vals),len(vals))
out=pd.DataFrame(rows); out.to_csv('scripts/miner_1_20340720_volatility_scaled_20d_momentum_ic.csv',index=False)
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_1_20340720_volatility_scaled_20d_momentum_signal.csv',index=False)
