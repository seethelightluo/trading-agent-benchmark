import pandas as pd,numpy as np,json,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
base=pd.read_csv('scripts/miner_3_20270325_vix_range_reversal_signal.csv',parse_dates=['date']).set_index('date')
sm=pd.read_csv('scripts/miner_1_20270325_smoothed_risk_reversal_signal.csv',parse_dates=['date']).set_index('date')
ys=pd.read_csv('scripts/miner_2_20270325_yieldspread_residual_signal.csv',parse_dates=['date']).set_index('date')
assets=list(base.columns); prices={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 prices[a]=d.close
p=pd.DataFrame(prices).sort_index().loc[:cut]
# Cross-sectional residual: remove contemporaneous linear exposures to two established reversal families.
out=[]
for dt in base.index:
 if dt not in sm.index or dt not in ys.index: continue
 z=pd.DataFrame({'x':base.loc[dt],'sm':sm.loc[dt],'ys':ys.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)<8: continue
 X=np.column_stack([np.ones(len(z)),z.sm.rank(pct=True),z.ys.rank(pct=True)])
 beta=np.linalg.lstsq(X,z.x.values,rcond=None)[0]
 res=z.x.values-X@beta
 out.append(pd.Series(res,index=z.index,name=dt))
fac=pd.DataFrame(out); fac.index.name='date'; fac=fac.reindex(columns=assets); fac.to_csv('scripts/miner_3_20270325_orthogonal_vix_reversal_signal.csv')
for h in [1,5,10]:
 fwd=p.pct_change(h).shift(-h); vals=[];ns=[]
 for dt in fac.index:
  if dt not in fwd.index: continue
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=pd.Series(vals); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
