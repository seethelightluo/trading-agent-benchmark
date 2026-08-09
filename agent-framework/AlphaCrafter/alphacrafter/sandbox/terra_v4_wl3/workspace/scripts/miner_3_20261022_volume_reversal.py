import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut='2026-10-21'
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
frames={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 frames[a]=d
P=pd.concat({a:d.close for a,d in frames.items()},axis=1).sort_index()
V=pd.concat({a:d.volume for a,d in frames.items()},axis=1).sort_index()
R=P.pct_change()
# volume-confirmed reversal: negative trailing return weighted by unusual volume, normalized per asset
vr=V.div(V.rolling(20,min_periods=10).median()).replace([np.inf,-np.inf],np.nan)
fac=-R.rolling(5,min_periods=5).sum()*np.log1p(vr.clip(upper=10))
fac.to_csv('scripts/miner_3_20261022_volume_reversal_signal.csv')
for h in [1,5,10]:
 fwd=P.pct_change(h).shift(-h); vals=[];ds=[];ns=[]
 for dt in fac.index:
  x=fac.loc[dt].dropna();y=fwd.loc[dt].reindex(x.index).dropna();x=x.reindex(y.index)
  if len(x)>=8 and x.nunique()>1 and y.nunique()>1: vals.append(spearmanr(x,y).statistic);ds.append(dt);ns.append(len(x))
 s=pd.Series(vals,index=pd.DatetimeIndex(ds)); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean(),fac.notna().sum(axis=1).mean()/15))
 if h==1: print('regimes',[(y,round(s[s.index.year==y].mean(),5),len(s[s.index.year==y])) for y in range(2020,2027)])
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
print('period',P.index.min(),P.index.max(),'assets',P.shape[1])
print('vol_valid',V.notna().mean().mean())
