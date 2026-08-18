import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:D[s]=get_index_daily_data(s,4000)
 except:D[s]=get_stock_daily_data(s,4000)
p=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill(); r=p.pct_change(); m=p.pct_change(40); down=(-r.clip(upper=0)).rolling(40).sum(); up=r.clip(lower=0).rolling(40).sum(); f=(m/(1+down/(up+1e-9))).shift(1); fw=p.shift(-10)/p-1; a=[]; n=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i],fw.iloc[i]],axis=1).dropna()
 if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]));n.append(len(z))
a=pd.Series(a);print('factor=downside_persistence_40d dates=%d avg_n=%.2f coverage=%.3f'%(len(a),np.mean(n),np.mean(n)/15));print('IC=%.6f ICIR=%.6f hit=%.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
for w in [365,730,1095]:q=a.tail(w);print('recent%d IC=%.6f ICIR=%.6f hit=%.4f'%(w,q.mean(),q.mean()/q.std(),(q>0).mean()))
