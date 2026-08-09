import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
fac={}; fwd={}; prices={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=cut].drop_duplicates('date').set_index('date')
 p=d.close.astype(float); r=p.pct_change()
 # All rolling operations remain on native asset observations.
 dn=r.where(r<0)
 down=dn.rolling(20,min_periods=10).std()
 fac[a]=(p.pct_change(20)/(np.sqrt(20)*down+0.002)).clip(-20,20)
 for h in (1,5,10): fwd.setdefault(h,{})[a]=p.shift(-h)/p-1
F=pd.DataFrame(fac).sort_index(); F.to_csv('scripts/miner_1_20270325_downside_momentum_native_signal.csv')
def run(h):
 Y=pd.DataFrame(fwd[h]).sort_index(); vals=[]; ds=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); ir=s.mean()/s.std(ddof=1)
 print(h,'dates',len(s),'avgN',round(np.mean(ns),3),'IC',round(s.mean(),6),'ICIR',round(ir,6),'hit',round((s>0).mean(),4)); return s
s1=run(1); run(5); run(10)
print('coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-03-24')]:
 q=s1[(s1.index>=lo)&(s1.index<=hi)]; print(name,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('signal_artifact scripts/miner_1_20270325_downside_momentum_native_signal.csv')
