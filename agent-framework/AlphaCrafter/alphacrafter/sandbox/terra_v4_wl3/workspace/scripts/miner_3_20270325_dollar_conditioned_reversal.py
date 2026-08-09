import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
# Dollar-regime-conditioned cross-sectional 3d reversal: stronger fade when lagged DXY move is extreme.
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date').set_index('date').close
F={}; P={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 p=d.close.reindex(dxy.index).ffill(); r=p.pct_change()
 dr=dxy.pct_change(3).abs(); scale=(dr/dr.rolling(60,min_periods=20).median()).shift(1).clip(0.5,2.5)
 # lag-safe: today's signal uses through prior close; scale is lagged one day
 f=-(p.pct_change(3))*scale
 F[a]=f; P[a]=p.pct_change(1).shift(-1)
fac=pd.DataFrame(F).sort_index(); fwd=pd.DataFrame(P).reindex(fac.index); fac.to_csv('scripts/miner_3_20270325_dollar_conditioned_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 # recompute forward from aligned prices via signal relation (daily factor vs h-day returns)
 vals=[]; ds=[]; ns=[]
 prices=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}).sort_index()
 fwd=prices.pct_change(h).shift(-h).reindex(fac.index)
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds)
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
