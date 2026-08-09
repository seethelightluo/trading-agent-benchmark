import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut='2026-11-04'; assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets},axis=1).sort_index().loc[:cut]
R=P.pct_change(); vol=R.rolling(20,min_periods=15).std()
# volatility-scaled relative strength: trailing 20d excess return versus cross-asset median, scaled by own risk
fac=P.pct_change(20).sub(P.pct_change(20).median(axis=1),axis=0).div(vol)
fac=fac.replace([np.inf,-np.inf],np.nan); fac.to_csv('scripts/miner_2_20261105_scaled_relative_strength_signal.csv')
for h in [1,5,10]:
 fwd=P.pct_change(h).shift(-h); vals=[];ds=[];ns=[]
 for dt in fac.index:
  x=fac.loc[dt].dropna();y=fwd.loc[dt].reindex(x.index).dropna();x=x.reindex(y.index)
  if len(x)>=8 and x.nunique()>1 and y.nunique()>1: vals.append(spearmanr(x,y).statistic);ds.append(dt);ns.append(len(x))
 s=pd.Series(vals,index=pd.DatetimeIndex(ds));print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean()))
 if h==1: print('regimes',[(y,round(s[s.index.year==y].mean(),5),len(s[s.index.year==y])) for y in range(2020,2027)])
print('coverage',fac.notna().sum(axis=1).mean()/15,'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean());print('period',P.index.min(),P.index.max(),'assets',P.shape[1])
