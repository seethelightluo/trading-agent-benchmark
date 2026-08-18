import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-01-09'); base='../persistent/stock_data'; close={}; vol={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); d=d[d.date<=cutoff].set_index('date').sort_index()
 close[s]=d.close.astype(float); vol[s]=d.volume.astype(float)
P=pd.DataFrame(close).sort_index(); V=pd.DataFrame(vol).reindex(P.index); r=P.pct_change()
# Contrarian short return, scaled by risk and strengthened by abnormal volume (all lagged).
rv=r.rolling(20,min_periods=15).std(); vshock=(V/V.rolling(20,min_periods=15).mean()).clip(0.25,4)
sig=(-(r.rolling(5,min_periods=5).sum())/(rv*np.sqrt(20)+1e-8)*np.log1p(vshock)).shift(1)
print('rows',len(P),'range',P.index.min().date(),P.index.max().date())
for h in [1,5,10,20]:
 fwd=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(z))
 a=np.array(vals); ic=a.mean(); ir=ic/(a.std(ddof=1)+1e-12)*np.sqrt(len(a))
 print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4))
 for n in [250,500]:
  if len(a)>=n:
   q=a[-n:]; print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(len(q)),6))
print('turnover_proxy',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'panel_valid',round(sig.notna().sum().sum()/sig.size,4))
