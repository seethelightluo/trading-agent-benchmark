import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close.rename(s)
p=pd.concat([load(s) for s in U],axis=1).sort_index(); dxy=load('DXY').reindex(p.index).ffill(); r=p.pct_change(); dr=dxy.pct_change()
beta=r.rolling(60,min_periods=40).cov(dr).div(dr.rolling(60,min_periods=40).var(),axis=0)
f=-(p.pct_change(60)-beta.shift(1).mul(dxy.pct_change(60),axis=0)).div(r.rolling(60,min_periods=40).std().shift(1)).shift(1)
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20321223_macro_residual_reversal_signal.csv',index=False)
for h in [10,20,40,60]:
 fr=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for i in range(len(f)):
  x=f.iloc[i]; y=fr.iloc[i]; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
 a=np.asarray(vals); print(h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'dates',len(a),'avgN',np.mean(ns))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'period',f.index.min(),f.index.max())
