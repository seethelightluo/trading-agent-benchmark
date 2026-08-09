import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); paths=glob.glob('../persistent/stock_data/*.csv'); assets=[os.path.basename(x)[:-4] for x in paths]
F={}; C={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); d=d[d.index<=cut]
 C[a]=d.close
 r=d.close.pct_change(); vol=r.rolling(20,min_periods=15).std()
 # short reversal, damp extreme observations via volatility normalization
 F[a]=(-r.rolling(3).sum()/vol.replace(0,np.nan)).clip(-5,5)
fac=pd.DataFrame(F).sort_index(); close=pd.DataFrame(C).reindex(fac.index)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date').close
vr=v.pct_change(5); vv=vr.rolling(60,min_periods=30).std(); shock=(vr/vv.replace(0,np.nan)).clip(-3,3)
# reversal stronger in VIX shocks, but leave calm regime at baseline
fac=fac.mul((1+0.35*shock.clip(lower=0)/3).reindex(fac.index),axis=0)
fac.to_csv('scripts/miner_1_20270325_vix_shock_shortrev_signal.csv')
for h in [1,5,10]:
 vals=[]; ns=[]; ds=[]
 y=close.pct_change(h).shift(-h)
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.nunique().min()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds); print('horizon',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'dates',len(q),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7))
print('coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'assets',len(assets))
