import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index()
r=p.pct_change(); med=r.median(axis=1); breadth=(r>0).sum(axis=1)/r.notna().sum(axis=1)
# cross-asset residual shock: fade 3d idiosyncratic move, normalized by 20d vol
res3=r.rolling(3).sum().sub(r.rolling(3).sum().median(axis=1),axis=0)
vol=r.rolling(20).std()
base=-res3/vol
variants={'base':base,'breadth_extreme':base.where((breadth<.30)|(breadth>.70)), 'stress_only':base.where(breadth<.30), 'dispersion_gate':base.where(r.std(axis=1)>r.std(axis=1).rolling(60).median())}
for name,f in variants.items():
 for h in [1,3,5,10]:
  vals=[]; ns=[]
  fr=r.rolling(h).sum()
  for i in range(40,len(p)-h):
   z=pd.concat([f.iloc[i-1],fr.iloc[i+h-1]],axis=1).dropna()
   if len(z)>=8:
    q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
    if np.isfinite(q): vals.append(q);ns.append(len(z))
  a=np.array(vals)
  print(name,'H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
# save base artifact
base.to_csv('scripts/miner_1_20310417_macro_residual_signal.csv',index_label='date')
