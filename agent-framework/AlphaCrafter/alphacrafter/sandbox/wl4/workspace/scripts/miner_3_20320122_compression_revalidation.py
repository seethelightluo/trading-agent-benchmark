import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=3200)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change()
# candidate: residual reversal, stronger only when short vol compressed relative to medium vol
R=p.pct_change(10); vol10=r.rolling(10).std(); vol40=r.rolling(40).std(); vol60=r.rolling(60).std()
res=R.sub(R.median(axis=1),axis=0)
compression=(vol60/vol10).clip(0.5,3.0)
f=(-res/vol40*compression).shift(1)
# forward non-overlapping-ish daily observations as requested
fr=p.shift(-10)/p-1
ics=[]; ns=[]; turnover=[]; prev=None
for dt in f.index:
    a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
        q=a.rank(pct=True); turnover.append(np.nan if prev is None else np.mean((q-prev).abs()))
        prev=q
ics=np.array(ics); turnover=np.array(turnover)
def stat(x): return (np.nanmean(x), np.nanmean(x)/(np.nanstd(x,ddof=1)/np.sqrt(np.sum(np.isfinite(x)))))
print('end',p.index.max().date(),'dates',len(ics),'avgN',np.mean(ns),'coverage',len(ics)/(len(f)-1),'IC/ICIR',stat(ics),'hit',np.mean(ics>0),'turnover',np.nanmean(turnover))
for days in [365,730,1095]:
 x=ics[-days:]; print('recent',days,stat(x), 'hit',np.mean(x>0))
# decay same signal with horizons
for h in [5,10,20,40]:
 ff=p.shift(-h)/p-1; xx=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: xx.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,stat(np.array(xx)))
