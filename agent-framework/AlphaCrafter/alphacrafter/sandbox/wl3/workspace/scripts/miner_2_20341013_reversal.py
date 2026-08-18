import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=4000)
    if x is not None and len(x):
        x=x[['date','close']].copy(); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index()
# 5-day short-term reversal, volatility scaled; all inputs through t
r5=np.log(p/p.shift(5)); vol20=np.log(p/p.shift(1)).rolling(20).std()
f=-(r5/vol20)
# neutralize market/common component cross-section each day (rank is interpretable and robust)
f=f.rank(axis=1,pct=True)
f=f.sub(f.mean(axis=1),axis=0)
f=f.shift(0)
fwd=p.shift(-10)/p-1
ics=[]; n=[]; turnover=[]
prev=None
for dt in f.index:
    a=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
    if len(a)>=8:
        ics.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); n.append(len(a))
        sig=f.loc[dt].rank(pct=True)
        if prev is not None: turnover.append(np.nanmean(np.abs(sig-prev)))
        prev=sig
z=pd.Series(ics).dropna()
def stat(x): return (len(x),x.mean(), x.mean()/x.std(ddof=1) if x.std(ddof=1)>0 else np.nan, (x>0).mean())
print('dates',len(z),'avg_n',np.mean(n),'coverage',np.mean(n)/15,'IC/ICIR/hit',stat(z),'turnover',np.nanmean(turnover))
for k in [120,252,504,1000]:
 print('recent',k,stat(z.tail(k)))
# decay same factor against 1,5,10,20 forward
for h in [1,5,10,20]:
 q=[]
 fw=p.shift(-h)/p-1
 for dt in f.index:
  a=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 print('decay',h,stat(pd.Series(q).dropna()))
print('last',p.index[-1])
