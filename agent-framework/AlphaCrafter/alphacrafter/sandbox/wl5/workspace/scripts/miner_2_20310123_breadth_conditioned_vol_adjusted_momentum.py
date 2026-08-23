import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-01-22'); base=Path('../persistent/stock_data')
def load(s):
 d=pd.read_csv(base/(s+'.csv')); d.date=pd.to_datetime(d.date); return pd.to_numeric(d.set_index('date').sort_index().close,errors='coerce')
P=pd.concat([load(s).rename(s) for s in U],axis=1).sort_index().loc[:cut]
R=P.pct_change(); breadth=R.gt(0).mean(axis=1).rolling(20,min_periods=15).mean()
# Momentum is rewarded in broad participation, while low participation reverses it.
raw=P.pct_change(20); vol=R.rolling(40,min_periods=25).std(); gate=(breadth-0.5)*2
f=(raw/vol).mul(gate,axis=0).shift(1); fw=P.shift(-10).div(P).sub(1)
rows=[]; prev=None; turns=[]; art=[]
for dt in f.index:
 x=f.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 xx=x[ok]; yy=y[ok]; rows.append((dt,xx.corr(yy),ok.sum())); q=xx.rank(pct=True)
 if prev is not None: turns.append((q-prev.reindex(q.index)).abs().mean())
 prev=q
 for s,val in xx.items(): art.append({'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(val)})
o=pd.DataFrame(rows,columns=['date','ic','n']).dropna(); z=o.ic
print('candidate breadth_conditioned_vol_adjusted_momentum_20d');print('dates',len(z),'meanN',o.n.mean(),'coverage',o.n.mean()/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',np.mean(turns),'period',o.date.min().date(),o.date.max().date())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-01-22')]:
 q=o[(o.date>=a)&(o.date<=b)]; print('regime',a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10,20]:
 fw=P.shift(-h).div(P).sub(1); vals=[]
 for dt in f.index:
  x=f.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(x[ok].corr(y[ok]))
 print('decay',h,np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1),len(vals))
pd.DataFrame(art).to_csv('scripts/miner_2_20310123_breadth_conditioned_vol_adjusted_momentum_signal.csv',index=False)
