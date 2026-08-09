import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=sorted([os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')])
# Observation-only DXY: all inputs end at prior completed session.
x=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date').set_index('date').close
x=x[x.index<=cut]; dx=np.log(x).diff(); xm=dx.rolling(60,min_periods=30).median(); xmad=(dx-xm).abs().rolling(60,min_periods=30).median(); stress=((dx-xm)/(xmad+1e-6)).clip(-3,3)
rets={}; closes={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); d=d[d.index<=cut]
 closes[a]=d.close; rets[a]=d.close.pct_change()
r=pd.DataFrame(rets).sort_index(); market=r.median(axis=1)
# rolling asset beta to common cross-asset move; residualizes the one-day shock.
F={}
for a in assets:
 beta=r[a].rolling(60,min_periods=30).cov(market)/market.rolling(60,min_periods=30).var()
 resid=r[a]-beta*market
 F[a]=(-resid*(1+0.35*stress.reindex(r.index).fillna(0).abs())).clip(-.2,.2)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_2_20270325_dxy_residual_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=pd.DataFrame({a:closes[a].pct_change(h).shift(-h) for a in assets}).reindex(fac.index)
 vals=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
