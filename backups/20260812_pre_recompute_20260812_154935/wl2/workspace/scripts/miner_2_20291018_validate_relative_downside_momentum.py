import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<250:d=get_index_daily_data(s,1800)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); w=60
rel=P.pct_change(w).sub(P.pct_change(w).mean(axis=1),axis=0); down=R.clip(upper=0).pow(2).rolling(w).mean().pow(.5)
sig=(-rel/(down+1e-8)).shift(1); fwd=P.pct_change(10).shift(-10)
rows=[]; ics=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c); rows.append({'date':dt,**sig.loc[dt].to_dict()})
a=np.asarray(ics); print('dates',len(a),'assets',len(P.columns),'avgN',len(U),'IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.6f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean([pd.Series(r).count()-1 for r in rows])/len(U),sig.diff().abs().mean().mean()))
for lab,cut in [('2027+',pd.Timestamp('2027-01-01')),('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
 q=np.array([v for v,d in zip(ics,sig.index) if d>=cut and d in sig.index[:len(ics)]])
 # use aligned dates robustly
 ds=[r['date'] for r in rows]; q=np.array([v for v,d in zip(ics,ds) if d>=cut]); print(lab,len(q),q.mean(),q.mean()/q.std(ddof=1))
pd.DataFrame(rows).to_csv('scripts/miner_2_20291018_relative_downside_momentum_signal.csv',index=False)
