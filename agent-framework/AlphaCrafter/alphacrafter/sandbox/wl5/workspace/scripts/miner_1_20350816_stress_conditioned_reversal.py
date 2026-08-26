import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').close.rename(s)
P=pd.concat([load(s) for s in U],axis=1,sort=True).sort_index().loc[:'2035-08-01']; R=P.pct_change(); med=R.median(axis=1)
# causal idiosyncratic returns and beta-neutral residuals
b=R.rolling(60,min_periods=40).cov(med).div(med.rolling(60,min_periods=40).var(),axis=0)
res=R-b.mul(med,axis=0); vol=res.rolling(20,min_periods=15).std()*np.sqrt(10)
base=-res.rolling(10,min_periods=8).sum().div(vol+1e-12)
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.reindex(P.index).ffill()
# causal stress z-score, clipped to avoid unstable tails; stress-conditioned reversal
vm=v.rolling(120,min_periods=60).median(); mad=(v-vm).abs().rolling(120,min_periods=60).median()
stress=((v-vm)/(1.4826*mad+1e-9)).clip(-3,3)
factor=base.mul(1+0.60*np.tanh(stress),axis=0)
print('assets',P.shape[1],'rows',len(P),'period',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; vals=[]; ns=[]; ds=[]; ranks=[]; prev=None; turns=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(c): vals.append(c); ns.append(len(z)); ds.append(dt); rr=z.iloc[:,0].rank(pct=True); ranks.append(rr)
   if h==10 and prev is not None: turns.append(np.mean(abs(rr-prev.reindex(rr.index))))
   if h==10: prev=rr
 x=np.array(vals); print('horizon',h,'dates',len(x),'meanN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(x.mean(),6),'ICIR',round(x.mean()/(x.std(ddof=1)+1e-12),6),'hit',round(np.mean(x>0),6))
 if h==10:
  dsi=pd.DatetimeIndex(ds)
  for a,bnd in [('2023','2025'),('2026','2028'),('2029','2031'),('2032','2035')]:
   w=x[(dsi>=a)&(dsi<=bnd)]; print('regime',a,bnd,'dates',len(w),'IC',round(w.mean(),6) if len(w) else None)
  print('turnover',round(np.mean(turns),6))
  factor.stack().rename('factor_value').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20350816_stress_conditioned_reversal_signal.csv',index=False)
