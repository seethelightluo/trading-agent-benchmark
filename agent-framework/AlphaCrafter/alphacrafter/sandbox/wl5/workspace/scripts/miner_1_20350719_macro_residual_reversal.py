import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').close.rename(s)
px=pd.concat([load(s) for s in assets],axis=1,sort=True).sort_index(); px=px.loc[:'2035-07-19']; ret=px.pct_change()
csmed=ret.median(axis=1); res=ret.sub(csmed,axis=0); res20=res.rolling(20).sum(); vol=ret.rolling(40).std()*np.sqrt(252)
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); vix=vix.set_index('date').close.reindex(px.index).ffill(); vc=vix.pct_change(20).clip(lower=-.5,upper=1.5).fillna(0).clip(lower=0); macro=1/(1+vc)
factor=-(res20.div(vol+1e-8)).mul(macro,axis=0)
def evaluate(h):
 fr=px.shift(-h)/px-1; vals=[]; ns=[]; turns=[]; prev=None
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)<8: continue
  vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); r=z.iloc[:,0].rank(pct=True)
  if prev is not None: turns.append(np.mean(abs(r-prev.reindex(r.index))))
  prev=r
 x=np.array(vals); return len(x),np.mean(ns),x.mean(),x.mean()/(x.std(ddof=1)+1e-12),np.mean(x>0),np.mean(turns)
for h in [5,10,20]: print('horizon',h,'dates meanN IC ICIR hit turnover',evaluate(h))
fr=px.shift(-10)/px-1; obs=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: obs.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
o=pd.DataFrame(obs,columns=['date','ic']).set_index('date')
for a,b in [('2023','2025'),('2026','2028'),('2029','2031'),('2032','2035')]:
 q=o.loc[a:b].ic; print('regime',a,b,'dates',len(q),'IC',q.mean())
print('rows',len(px),'assets',px.shape[1],'coverage',factor.notna().stack().mean())
sig=factor.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_1_20350719_macro_residual_reversal_signal.csv',index=False)
