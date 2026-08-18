import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2029-11-14'); base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); d=d[d.date<=cutoff].set_index('date').sort_index(); px[s]=d.close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v[v.date<=cutoff].set_index('date').sort_index().close.astype(float).reindex(P.index).ffill()
# Stress-gated medium-term reversal: recent losses are more likely to mean-revert when volatility is elevated.
vz=((v-v.rolling(60,min_periods=30).mean())/(v.rolling(60,min_periods=30).std()+1e-8)).clip(-3,3)
down=np.sqrt((r.clip(upper=0)**2).rolling(20,min_periods=15).mean())*np.sqrt(20)
sig=(-P.pct_change(10)/(down+1e-8)).mul((1+0.6*np.maximum(vz,0)),axis=0).shift(1)
print('rows',len(P),'range',P.index.min().date(),P.index.max().date(),'vix_valid',round(v.notna().mean(),4))
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
   q=a[-n:]; print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(n),6))
print('turnover_proxy',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'panel_valid',round(sig.notna().sum().sum()/sig.size,4))
