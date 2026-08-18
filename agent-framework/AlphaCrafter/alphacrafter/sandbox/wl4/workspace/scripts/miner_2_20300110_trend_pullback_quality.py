import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-01-09'); base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); d=d[d.date<=cutoff].set_index('date').sort_index(); px[s]=d.close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Pullbacks are favored only when the medium-term trend is positive; continuous trend gate
mom60=P.pct_change(60); gate=(mom60.clip(lower=0)/(mom60.abs()+0.10)).clip(0,1)
sig=(-r.rolling(5,min_periods=5).sum()*gate).shift(1)
print('rows',len(P),'range',P.index.min().date(),P.index.max().date(),'assets',len(U))
for h in [1,5,10,20]:
 fwd=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(z))
 a=np.array(vals); ic=a.mean(); ir=ic/(a.std(ddof=1)+1e-12)*np.sqrt(len(a))
 print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'minN',min(ns),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4))
 print(' decay_recent250', round(a[-250:].mean(),6), round(a[-250:].mean()/(a[-250:].std(ddof=1)+1e-12)*np.sqrt(min(250,len(a))),6)) if len(a)>=250 else None
print('turnover_proxy',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'panel_valid',round(sig.notna().sum().sum()/sig.size,4))
