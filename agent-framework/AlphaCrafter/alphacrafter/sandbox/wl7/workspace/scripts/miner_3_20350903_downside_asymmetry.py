import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-09-02']
r=C.pct_change(); m=r.mean(axis=1)
# trailing beta residuals, all inputs lagged by one completed day
mu=m.rolling(60,min_periods=40).mean(); vr=m.rolling(60,min_periods=40).var()
cov=r.mul(m,axis=0).rolling(60,min_periods=40).mean()-r.rolling(60,min_periods=40).mean().mul(mu,axis=0)
beta=cov.div(vr,axis=0); e=r.sub(beta.mul(m,axis=0),axis=0)
# downside-asymmetry shock: recent negative residual path weighted more than positive path,
# normalized by downside residual deviation; positive score means stronger rebound candidate.
neg=e.clip(upper=0); pos=e.clip(lower=0)
neg20=neg.rolling(20,min_periods=15).sum(); pos20=pos.rolling(20,min_periods=15).sum()
down=neg.rolling(40,min_periods=25).std();
F=(-(neg20+0.35*pos20)/down).replace([np.inf,-np.inf],np.nan)
F=F.sub(F.median(axis=1),axis=0).shift(1)
def ev(h):
 y=C.shift(-h)/C-1; aa=[]; ds=[]; ns=[]
 for d in F.index:
  q=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8: aa.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(d);ns.append(len(q))
 a=np.asarray(aa);ds=np.asarray(ds); ir=a.mean()/a.std(ddof=1)*np.sqrt(252)
 print(f'H{h} IC {a.mean():.8f} ICIR {ir:.8f} dates {len(a)} avgN {np.mean(ns):.2f} hit {np.mean(a>0):.4f}')
 for lo,hi in [('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-09-02')]:
  z=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))]; print(' ',lo,len(z),f'IC {z.mean():.8f}' if len(z) else 'none')
 return a,ds
for h in [1,5,10,20]:ev(h)
print('coverage',F.notna().sum().sum()/(15*len(F)),'turnover',np.nanmean(np.abs(F.diff()).mean(axis=1)))
F.to_csv('scripts/miner_3_20350903_downside_asymmetry_signal.csv',index_label='date')
