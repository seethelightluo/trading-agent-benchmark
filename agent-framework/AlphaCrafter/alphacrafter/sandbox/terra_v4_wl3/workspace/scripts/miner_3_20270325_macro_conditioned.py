import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
# Macro-conditioned short reversal: fade 5d return in stressed (VIX above 60d median), follow 20d momentum otherwise.
px={};
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); px[a]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index(); r5=p.pct_change(5); mom=p.pct_change(20)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index(); vc=v['close'].reindex(p.index).ffill(); stress=(vc>vc.rolling(60,min_periods=30).median()).astype(float)
fac=(-r5*stress.values[:,None] + mom*(1-stress.values[:,None]))
fac.to_csv('scripts/miner_3_20270325_macro_conditioned_signal.csv')
vals=[]; ds=[]; ns=[]
fwd=p.pct_change(1).shift(-1)
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
s=pd.Series(vals,index=ds)
print('assets',len(assets),'rows',len(fac),'dates',len(s),'avgN',np.mean(ns),'coverage',fac.notna().sum(axis=1).mean()/len(assets))
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
for h in [5,10]:
 ff=p.pct_change(h).shift(-h); vv=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vv.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('H',h,'IC %.6f ICIR %.6f'%(np.mean(vv),np.mean(vv)/np.std(vv,ddof=1)))
