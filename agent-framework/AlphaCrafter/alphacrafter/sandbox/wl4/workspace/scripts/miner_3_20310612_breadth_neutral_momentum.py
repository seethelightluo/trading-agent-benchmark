import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:D[s]=get_index_daily_data(s,days=4000)
 except:D[s]=get_stock_daily_data(s,days=4000)
P=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=P.pct_change(); rel=P.pct_change(20).sub(P.pct_change(20).median(axis=1),axis=0); f=(rel/r.rolling(20).std()).shift(1)
for H in [1,5,10,20]:
 a=[];ns=[]; fw=P.shift(-H)/P-1
 for i in range(len(P)-H):
  z=pd.concat([f.iloc[i],fw.iloc[i]],axis=1).replace([np.inf,-np.inf],np.nan).dropna();ns.append(len(z))
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(a);print('h=%d dates=%d avg_n=%.2f coverage=%.4f IC=%.6f ICIR=%.6f hit=%.4f recent365=%.6f/%.6f'%(H,len(a),np.mean(ns),np.mean(ns)/15,a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),a[-365:].mean(),a[-365:].mean()/a[-365:].std(ddof=1)))
print('turnover=%.6f'%np.nanmean([f.iloc[i].rank(pct=True).sub(f.iloc[i-1].rank(pct=True)).abs().mean() for i in range(1,len(f))]))
