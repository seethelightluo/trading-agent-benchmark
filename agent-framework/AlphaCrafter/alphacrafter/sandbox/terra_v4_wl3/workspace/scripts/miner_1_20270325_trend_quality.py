import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change(1); mom=close.pct_change(20)
# Trend quality: 20d momentum penalized by recent 20d realized volatility; cross-sectional rank is used by evaluator.
vol=r.rolling(20,min_periods=15).std(); fac=mom/(vol.replace(0,np.nan)*np.sqrt(20)); fac=fac.replace([np.inf,-np.inf],np.nan); fac.to_csv('scripts/miner_1_20270325_trend_quality_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vs=[];ns=[];ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:vs.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ns.append(len(x));ds.append(dt)
 return pd.Series(vs,index=ds),ns
print('assets',len(C),'rows',len(fac))
for h in [1,5,10]:
 s,n=ev(h);print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
