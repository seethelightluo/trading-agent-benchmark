import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-08-19']
r=C.pct_change(); m=r.mean(axis=1)
# Rolling beta to the contemporaneous equal-weight cross-asset benchmark; residual trend is formed only from trailing data.
rm=m.rolling(60,min_periods=40); cov=r.mul(m,axis=0).rolling(60,min_periods=40).mean()-r.rolling(60,min_periods=40).mean().mul(m.rolling(60,min_periods=40).mean(),axis=0)
beta=cov.div(m.rolling(60,min_periods=40).var(),axis=0)
resid=r.sub(beta.mul(m,axis=0),axis=0)
res20=resid.rolling(20,min_periods=15).sum(); rv=resid.rolling(40,min_periods=25).std()
F=-(res20/rv).replace([np.inf,-np.inf],np.nan)
F=F.sub(F.median(axis=1),axis=0).shift(1)
def evalh(h):
 y=C.shift(-h)/C-1; A=[]; ds=[]; ns=[]
 for d in F.index:
  q=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:A.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ds.append(d); ns.append(len(q))
 a=np.asarray(A); ds=np.asarray(ds)
 ir=a.mean()/a.std(ddof=1)*np.sqrt(252) if len(a)>1 and a.std(ddof=1)>0 else np.nan
 print('H',h,'IC %.8f ICIR %.8f dates %d avgN %.2f hit %.4f'%(a.mean(),ir,len(a),np.mean(ns),np.mean(a>0)))
 for st,en in [('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-08-19')]:
  q=a[(ds>=pd.Timestamp(st))&(ds<=pd.Timestamp(en))]; print(' regime',st,len(q),'IC %.8f hit %.4f'%(q.mean() if len(q) else np.nan,np.mean(q>0) if len(q) else np.nan))
 return a,ds,ns
for h in [1,5,10,20]:evalh(h)
print('coverage %.4f turnover %.4f'%(F.notna().sum().sum()/(15*len(F)),np.nanmean(np.abs(F.diff()).mean(axis=1))))
F.to_csv('scripts/miner_3_20350820_beta_residual_reversal_signal.csv',index_label='date')
