import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
cl={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for a in assets}
cl=pd.DataFrame(cl).loc[:cut]
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:cut]
threshold=vix.rolling(60,min_periods=30).median().shift(1)
stress=(vix.shift(1)>threshold).reindex(cl.index).ffill().fillna(False).astype(bool)
mom=cl.pct_change(5).shift(1)
fac=mom.copy()
fac.loc[stress,:]=-mom.loc[stress,:]
fac=fac.sub(fac.median(axis=1),axis=0)
fac.to_csv('scripts/miner_1_20270325_vix_regime_momentum_signal.csv')
fwd=cl.pct_change(1).shift(-1); vals=[];ds=[];ns=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
s=pd.Series(vals,index=ds)
print('assets',len(assets),'dates',len(s),'avgN',round(np.mean(ns),2),'period',s.index.min(),s.index.max())
print('IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=a)&(s.index<=b)];print('regime',a,b,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [5,10]:
 fw=cl.pct_change(h).shift(-h); vv=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vv.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(vv);print('H',h,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
