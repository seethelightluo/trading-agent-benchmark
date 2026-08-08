import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in syms:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); d[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(d).sort_index().loc[:'2034-08-30']
ret=np.log(px).diff()
# lagged 10-session reversal: values on t use close through t, evaluated to t+1
sig=-ret.rolling(10,min_periods=10).sum().shift(0)
fwd=ret.shift(-1)
ics=[]; ns=[]; turns=[]
prev=None
for dt in sig.index:
 a=sig.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(a[ok],b[ok]).statistic); ns.append(ok.sum())
  ranks=a.rank(pct=True); turns.append(np.nan if prev is None else np.mean(abs(ranks-prev)))
  prev=ranks
ics=np.array(ics); turns=np.array(turns)
print('factor=negative 10d log return; dates',len(ics),'meanN',np.mean(ns),'coverage',np.mean(ns)/15)
print('H1 IC %.6f ICIR %.6f hit %.4f turnover10proxy %.6f'%(np.nanmean(ics),np.nanmean(ics)/np.nanstd(ics,ddof=1),np.mean(ics>0),np.nanmean(turns)))
for h in [1,5,10,20]:
 z=[]
 for dt in sig.index:
  a=sig.loc[dt]; b=ret.shift(-h).rolling(h).sum().loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(spearmanr(a[ok],b[ok]).statistic)
 z=np.array(z);print('H',h,'IC %.6f ICIR %.6f dates %d'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),len(z)))
for y in range(2020,2035):
 z=[v for dt,v in zip(sig.index,ics) if dt.year==y]
 if z: print(y,'%.5f %.5f'%(np.mean(z),np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan))
