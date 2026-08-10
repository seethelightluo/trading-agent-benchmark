import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[s]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Novel interpretable factor: medium-term risk-adjusted momentum, gated to avoid
# buying assets in a broad downtrend. Rank cross-sectionally each day.
mom=close.pct_change(20); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
raw=mom/vol.replace(0,np.nan)
breadth=(mom>0).mean(axis=1)
# activation rises from 40% to 70% positive breadth; defensive reversal is not used
act=((breadth-0.40)/0.30).clip(0,1)
z=raw.sub(raw.mean(axis=1),axis=0).div(raw.std(axis=1).replace(0,np.nan),axis=0)
fac=z.mul(act,axis=0)
fac.to_csv('scripts/miner_3_20270325_gated_risk_momentum_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); out=[]; ns=[]; ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   out.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x)); ds.append(dt)
 return pd.Series(out,index=ds),ns
print('assets',len(C),'rows',len(fac),'period',fac.index.min().date(),fac.index.max().date())
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean(),'active_dates',(act>0).sum())
# rough library similarity to saved signal artifacts
for p in glob.glob('scripts/*signal.csv'):
 try:
  old=pd.read_csv(p,index_col=0,parse_dates=True); common=fac.index.intersection(old.index); cols=fac.columns.intersection(old.columns)
  if len(common)>100 and len(cols)>=8:
   rho=fac.loc[common,cols].stack().corr(old.loc[common,cols].stack())
   if abs(rho)>.7: print('similar',p,round(rho,4))
 except Exception: pass
