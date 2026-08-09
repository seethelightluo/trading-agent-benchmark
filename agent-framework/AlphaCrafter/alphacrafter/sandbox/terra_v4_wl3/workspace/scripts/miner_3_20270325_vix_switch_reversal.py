import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); root='../persistent/stock_data'; assets=[os.path.basename(x)[:-4] for x in glob.glob(root+'/*.csv')]
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index(); v=v[v.index<=cut].close
# Lag all macro state: at date t use VIX through t-1.
state=(v.shift(1)>v.shift(1).rolling(60,min_periods=30).median()).astype(float)
F={}; fw={h:{} for h in [1,5,10]}
for a in assets:
 d=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change(3)
 # Fade 3d moves in elevated-VIX regime; follow 3d trend in calm regime.
 F[a]=(-r*(2*state.reindex(d.index)-1)).clip(-.2,.2)
 for h in fw: fw[h][a]=d.close.pct_change(h).shift(-h)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_3_20270325_vix_switch_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in fw:
 fwd=pd.DataFrame(fw[h]).reindex(fac.index); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),fac.notna().sum(axis=1).mean()/len(assets)))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
