import pandas as pd, numpy as np, os, glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
base='../persistent/stock_data'; idx='../persistent/index_data'
assets=sorted(os.path.basename(x)[:-4] for x in glob.glob(base+'/*.csv'))
px={}
for a in assets:
 d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).sort_values('date').set_index('date'); px[a]=d.close[d.index<=cut]
# observation-only VIX; macro state is known at date t, no forward leakage
m=pd.read_csv(idx+'/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date').close
m=m[m.index<=cut].reindex(pd.date_range(min(x.index.min() for x in px.values()),cut)).ffill()
vixret=m.pct_change(5); vixmed=m.rolling(60,min_periods=30).median()
# candidate: short reversal favored when VIX is elevated vs trailing median, neutral otherwise
F={}
for a,c in px.items():
 r=c.pct_change(); vol=r.rolling(10,min_periods=8).std()
 raw=(-r/vol)
 # macro multiplier: 1 + elevated VIX indicator, interpretable and nonnegative
 state=(vixmed>0) # placeholder alignment
 F[a]=raw.reindex(m.index)*(1.0+(m>vixmed).astype(float))
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_2_20270325_vix_gated_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt], 'y':pd.Series({a:px[a].pct_change(h).shift(-h).get(dt) for a in assets})}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds)
 print('H',h,'dates',len(s),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('coverage %.4f turnover %.4f'%(fac.notna().sum(axis=1).mean()/len(assets),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
