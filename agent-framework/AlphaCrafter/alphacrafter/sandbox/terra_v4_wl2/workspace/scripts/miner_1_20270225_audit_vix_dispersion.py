import pandas as pd, numpy as np, os
files=['../persistent/factor_signals_miner_1_20270225_vix_dispersion_reversal10.csv','../persistent/factor_signals_miner_3_20270225_continuous_regime_reversal.csv','../persistent/factor_signals_miner_2_20270225_breadth50_reversal3.csv']
xs=[]
for p in files:
 if os.path.exists(p):
  d=pd.read_csv(p); d['key']=d.date.astype(str)+'|'+d.symbol.astype(str); xs.append((p,d.set_index('key').signal))
for i,(p,a) in enumerate(xs):
 print('\n',os.path.basename(p),'finite',a.notna().sum(),'nonzero',((a.fillna(0)!=0)).sum(),'uniq',a.dropna().nunique())
 for q,b in xs[i+1:]:
  z=pd.concat([a,b],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  print('vs',os.path.basename(q),'aligned',len(z),'rho',z.iloc[:,0].corr(z.iloc[:,1]) if len(z)>2 else np.nan,'exact',np.allclose(z.iloc[:,0],z.iloc[:,1]))
