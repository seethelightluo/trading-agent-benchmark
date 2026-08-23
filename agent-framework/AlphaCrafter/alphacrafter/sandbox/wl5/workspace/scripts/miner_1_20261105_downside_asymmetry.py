import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2026-11-04'; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')
 r=x.close.pct_change(fill_method=None)
 # downside asymmetry: fraction of realized variance from negative returns, inverse is defensive quality
 neg=r.where(r<0,0.0)
 down=neg.pow(2).rolling(30,min_periods=20).mean()
 tot=r.pow(2).rolling(30,min_periods=20).mean()
 sig=-(down/(tot+1e-12)).clip(0,1) # higher means fewer downside shocks
 D[s]=pd.DataFrame({'sig':sig,'r1':x.close.pct_change().shift(-1),'r5':x.close.pct_change(5).shift(-5),'r10':x.close.pct_change(10).shift(-10)})
all_dates=sorted(set().union(*[set(z.index) for z in D.values()]))
for h in ['r1','r5','r10']:
 vals=[]; dates=[]; ns=[]
 for dt in all_dates:
  z=pd.DataFrame({'x':{s:D[s].sig.get(dt,np.nan) for s in U},'y':{s:D[s][h].get(dt,np.nan) for s in U}}).dropna()
  if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.x,z.y).statistic);dates.append(dt);ns.append(len(z))
 q=pd.Series(vals,index=pd.to_datetime(dates)); print(h,'dates',len(q),'avg_n',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 for lab,a,b in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025-26','2025','2026')]:
  z=q[(q.index>=a)&(q.index<=b+'-12-31')];print(lab,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
S=pd.DataFrame({s:D[s].sig for s in U}); print('coverage',round(S.notna().mean().mean(),4),'turnover',round(S.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'period',S.index.min().date(),S.index.max().date())
# artifact for potential admission
S.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20261105_downside_asymmetry_signal.csv',index=False)
