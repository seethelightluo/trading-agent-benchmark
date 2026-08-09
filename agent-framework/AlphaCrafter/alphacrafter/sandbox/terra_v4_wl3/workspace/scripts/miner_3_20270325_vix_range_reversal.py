import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={};P={}
# Observation-only VIX is used only as conditioning, never traded.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date')
vc=vix['close'].reindex(pd.date_range('2020-01-01',cut,freq='D')).ffill()
vp=vc.rolling(252,min_periods=60).rank(pct=True)
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date')
 r=d.close.pct_change(); rv=r.rolling(15,min_periods=10).std()
 rng=(d.high-d.low)/d.close
 expansion=(rng.rolling(5,min_periods=3).mean()/(rng.rolling(30,min_periods=15).median()+1e-8)-1).clip(lower=0,upper=3)
 # Fade 3-day overreaction, with stronger weight when both asset range and broad VIX regime expand.
 cond=(1+0.5*expansion)*(1+0.35*vp.reindex(d.index).fillna(0.5))
 F[a]=(-r.rolling(3).sum()/(rv*np.sqrt(3)+1e-8)*cond).clip(-12,12); P[a]=d.close
fac=pd.DataFrame(F).sort_index(); p=pd.DataFrame(P).sort_index(); out='scripts/miner_3_20270325_vix_range_reversal_signal.csv'; fac.to_csv(out)
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=p.pct_change(h).shift(-h); vals=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
