import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
    a=os.path.basename(p)[:-4]
    d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
    C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index()
# One-day reversal is activated only when cross-asset dispersion is elevated.
r1=close.pct_change(1)
med=r1.median(axis=1)
rel=r1.sub(med,axis=0)
z=rel.sub(rel.mean(axis=1),axis=0).div(rel.std(axis=1).replace(0,np.nan),axis=0)
disp=r1.std(axis=1)
threshold=disp.rolling(60,min_periods=30).median()
active=(disp>threshold).astype(float)
fac=-z.mul(active,axis=0)
fac.to_csv('scripts/miner_2_20270325_dispersion1_reversal_signal.csv')
def evaluate(h):
    y=close.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
    for dt in fac.index:
        x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
        if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
            vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x)); dates.append(dt)
    s=pd.Series(vals,index=dates)
    return s,ns
print('assets',len(C),'rows',len(fac),'active_dates',int(active.sum()))
for h in [1,5,10]:
 s,n=evaluate(h)
 print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]
   print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
