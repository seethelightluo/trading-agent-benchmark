import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'))
 d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
 px[s]=d['close']
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Conditional volatility-scaled short reversal: recent 3d reversal divided by 10d vol,
# activated only when cross-asset dispersion is elevated (above its trailing 60d median).
disp=R.std(axis=1)
active=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
sig=-(P.pct_change(3))/R.rolling(10,min_periods=8).std()
sig=sig.where(active.astype(bool), np.nan)
fwd=P.pct_change(1).shift(-1)
rows=[]
for dt in sig.index:
 a=sig.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'median_n',x.n.median(),'coverage',x.n.sum()/(len(x)*15))
for label,y in [('all',x),('2020-22',x.loc['2020':'2022']),('2023-25',x.loc['2023':'2025']),('2026-27',x.loc['2026':'2027']),('2028-31',x.loc['2028':'2031'])]:
 if len(y): print(label,'meanIC',y.ic.mean(),'ICIR',y.ic.mean()/y.ic.std(ddof=1),'hit', (y.ic>0).mean(),'n_dates',len(y))
# decay horizons
for h in [1,3,5,10]:
 ff=P.pct_change(h).shift(-h); rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'IC',np.nanmean(rr),'ICIR',np.nanmean(rr)/np.nanstd(rr,ddof=1),'obs',len(rr))
# signal artifact
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20310220_dispersion_volreversal_signal.csv',index=False)

