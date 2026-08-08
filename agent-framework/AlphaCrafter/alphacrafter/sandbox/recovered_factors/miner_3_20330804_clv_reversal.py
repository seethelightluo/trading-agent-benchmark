import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
from pathlib import Path
files=glob.glob('../persistent/stock_data/*.csv')
assets={Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in files}
# one interpretable idea: multi-day close-location pressure (CLV), reversed; OHLC only
clv={}
for a,d in assets.items():
    den=(d.high-d.low).replace(0,np.nan)
    x=((2*d.close-d.high-d.low)/den).rolling(5,min_periods=4).mean()
    clv[a]=-x.shift(1) # lag, low close-location => positive reversal score
sig=pd.DataFrame(clv); close=pd.DataFrame({a:d.close for a,d in assets.items()}).sort_index()
# align and calculate forward returns by calendar shared dates
sig=sig.reindex(close.index)
for h in [1,5,10,20]:
    fr=close.shift(-h)/close-1
    ics=[]; dates=[]; nvals=[]
    for dt in sig.index:
        z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); nvals.append(len(z))
    x=np.array(ics); print('H',h,'dates',len(x),'meanN',np.mean(nvals),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0))
# coverage and turnover
print('assets',len(assets),'date range',sig.index.min(),sig.index.max(),'coverage',sig.notna().sum().sum()/(sig.shape[0]*sig.shape[1]))
r=sig.rank(axis=1,pct=True); print('turnover10',((r-r.shift(10)).abs().mean(axis=1).mean()))
# regimes h10
fr=close.shift(-10)/close-1
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 x=[]
 for dt in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('regime',lo,hi,len(x),x.mean() if len(x) else np.nan,(x.mean()/x.std(ddof=1)) if len(x)>1 else np.nan)
