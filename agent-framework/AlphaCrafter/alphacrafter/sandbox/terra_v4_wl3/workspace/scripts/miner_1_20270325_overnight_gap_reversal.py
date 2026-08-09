import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
data={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 data[a]=d[d.index<=cut]
# Overnight gap reversal: fade gap from prior close to today's open, normalized by recent gap volatility.
opens=pd.DataFrame({a:x['open'] for a,x in data.items()}).sort_index()
closes=pd.DataFrame({a:x['close'] for a,x in data.items()}).sort_index()
gap=opens/closes.shift(1)-1
gapvol=gap.rolling(20,min_periods=10).std()
fac=-(gap/(gapvol+1e-8))
# cross-sectional robust winsorization and centering
fac=fac.sub(fac.median(axis=1),axis=0)
lo=fac.quantile(.05,axis=1); hi=fac.quantile(.95,axis=1)
fac=fac.clip(lower=lo,upper=hi,axis=0)
fac.to_csv('scripts/miner_1_20270325_overnight_gap_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=closes.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds)
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
