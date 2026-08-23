import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in A:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float)
 px[s]=d.sort_index()
p=pd.concat(px,axis=1).sort_index().loc[:'2032-10-27']; r=p.pct_change()
# A pullback is more attractive when short-term volatility is compressed versus its 60d baseline.
# High score = negative recent return, scaled by compression ratio.
v10=r.rolling(10,min_periods=8).std(); v60=r.rolling(60,min_periods=40).std()
sig=-(p.pct_change(10))*v10.div(v60).replace(0,np.nan)
def calc(h):
 f=p.shift(-h).div(p)-1; out=[]; ns=[]; dates=[]
 for i,dt in enumerate(p.index[:-h]):
  z=pd.concat([sig.iloc[i].rename('x'),f.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.x,z.y).statistic);ns.append(len(z));dates.append(dt)
 x=np.asarray(out); return len(x),float(np.mean(ns)),float(np.mean(x)),float(np.mean(x)/np.std(x,ddof=1)),float(np.mean(x>0)),float(np.mean(np.asarray(ns)/15)),pd.Series(x,index=dates)
print('universe',len(A),'data_dates',len(p),'signal_coverage',float(sig.notna().stack().mean()))
for h in [5,10,20,40]:
 n,avg,ic,ir,hit,cov,series=calc(h); print('horizon',h,'dates',n,'avg_n',round(avg,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4),'coverage',round(cov,4)); print('regimes',series.groupby(series.index.year).mean().round(5).to_dict())
# rank turnover proxy
u=[]
for i in range(1,len(sig)):
 z=pd.concat([sig.iloc[i-1].rank(pct=True),sig.iloc[i].rank(pct=True)],axis=1).dropna()
 if len(z):u.append(float(np.mean(abs(z.iloc[:,1]-z.iloc[:,0]))))
print('turnover',float(np.mean(u)),'max_abs_library_correlation','not_computed')
