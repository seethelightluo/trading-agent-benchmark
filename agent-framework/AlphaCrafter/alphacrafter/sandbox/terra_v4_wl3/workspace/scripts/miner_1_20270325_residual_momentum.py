import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
    a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
    C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); ret=close.pct_change()
# Beta-neutral medium-term momentum: 20d return unexplained by the equal-weight tradable benchmark.
bench=ret.mean(axis=1)
var=bench.rolling(60,min_periods=40).var()
cov=ret.apply(lambda x: x.rolling(60,min_periods=40).cov(bench))
beta=cov.div(var.replace(0,np.nan),axis=0)
resid=ret.sub(beta.mul(bench,axis=0),axis=0)
fac=resid.rolling(20,min_periods=15).sum()
fac=fac.replace([np.inf,-np.inf],np.nan); fac.to_csv('scripts/miner_1_20270325_residual_momentum_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x)); dates.append(dt)
 s=pd.Series(vals,index=dates); return s,ns
print('assets',len(C),'rows',len(fac))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
  q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
