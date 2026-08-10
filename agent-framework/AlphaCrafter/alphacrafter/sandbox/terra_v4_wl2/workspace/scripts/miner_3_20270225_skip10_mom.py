import pandas as pd,numpy as np
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in symbols:
 P[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:'2027-02-25']
px=pd.DataFrame(P); fac=px.shift(10)/px.shift(1)-1
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; ns=[]; turns=[]
 for dt in fac.index:
  a,b=fac.loc[dt],fwd.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(a[ok],b[ok]).statistic);ns.append(ok.sum())
   old=fac.shift(10).loc[dt]; ko=ok&old.notna()
   if ko.sum()>=8: turns.append(np.mean(np.sign(a[ko])!=np.sign(old[ko])))
 x=np.array(vals);print(h,'dates',len(x),'n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'turn',round(np.mean(turns),4))
for y in range(2020,2028):
 fwd=px.shift(-5)/px-1;x=[]
 for dt in fac.index:
  if dt.year==y:
   a,b=fac.loc[dt],fwd.loc[dt];ok=a.notna()&b.notna()
   if ok.sum()>=8:x.append(spearmanr(a[ok],b[ok]).statistic)
 print('year',y,'dates',len(x),'IC5',round(np.mean(x),6) if x else None)
print('coverage',round(fac.notna().mean().mean(),4),'matrix',round(fac.notna().sum().sum()/fac.size,4))
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_skip10_mom.csv',index=False)
