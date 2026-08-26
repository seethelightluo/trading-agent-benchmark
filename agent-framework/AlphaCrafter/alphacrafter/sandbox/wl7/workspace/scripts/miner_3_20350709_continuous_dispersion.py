import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-07-08']
r=C.pct_change(); vol=r.rolling(20,min_periods=15).std(); disp=vol.median(axis=1)
# Continuous dispersion impulse: residual 5d reversal, volatility scaled, weighted by
# the positive standardized increase in cross-asset dispersion. All inputs lagged one day.
raw=-(C/C.shift(5)-1); raw=raw.sub(raw.median(axis=1),axis=0)
base_sig=raw/vol
z=(disp-disp.rolling(60,min_periods=40).median())/(disp.rolling(60,min_periods=40).std())
imp=(disp-disp.shift(5))/(disp.rolling(60,min_periods=40).std())
weight=(1+np.maximum(z,0))*(1+np.maximum(imp,0))
f=base_sig.mul(weight,axis=0).shift(1)
# compare unconditional and a smooth bounded version
f2=base_sig.mul(np.tanh(z.clip(lower=0)+imp.clip(lower=0)),axis=0).shift(1)
ybase=C.shift(-20)/C-1

def report(name,F):
 a=[];ds=[];ns=[]
 for d in F.index:
  q=pd.concat([F.loc[d],ybase.loc[d]],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(d);ns.append(len(q))
 a=np.asarray(a);ds=np.asarray(ds)
 print(name,'IC %.8f ICIR %.8f dates %d avgN %.2f hit %.4f coverage %.4f'%(a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),len(a),np.mean(ns),np.mean(a>0),F.notna().sum().sum()/(15*len(F))))
 for st,en in [('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-07-08')]:
  q=a[(ds>=pd.Timestamp(st))&(ds<=pd.Timestamp(en))]; print(' regime',st,len(q),round(q.mean(),8) if len(q) else None)
 for h in [1,5,10,20]:
  yy=C.shift(-h)/C-1; aa=[]
  for d in F.index:
   q=pd.concat([F.loc[d],yy.loc[d]],axis=1).dropna()
   if len(q)>=8:aa.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
  aa=np.asarray(aa);print(' H',h,'IC %.8f ICIR %.8f dates %d'%(aa.mean(),aa.mean()/aa.std(ddof=1)*np.sqrt(252),len(aa)))
report('continuous_unbounded',f); report('continuous_bounded',f2)
f.to_csv('scripts/miner_3_20350709_continuous_dispersion_signal.csv',index_label='date')
