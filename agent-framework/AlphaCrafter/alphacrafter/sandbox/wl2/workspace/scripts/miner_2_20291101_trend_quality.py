import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<250: d=get_index_daily_data(s,1800)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); w=60
ret=P.pct_change(w); vol=R.rolling(w).std()*np.sqrt(w); breadth=(R>0).rolling(w).mean()
sig=(-(ret/(vol+1e-8))*breadth).shift(1); fwd=P.pct_change(10).shift(-10)
rows=[]; vals=[]; dates=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append(c); dates.append(dt); rows.append({'date':dt,**sig.loc[dt].to_dict()})
a=np.asarray(vals); print('universe',len(P.columns),'dates',len(a),'avgN',np.mean([pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna().shape[0] for d in dates]))
print('h10 IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.6f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),sig.notna().mean().mean(),sig.diff().abs().mean().mean()))
for h in [1,3,5,10,20]:
 ff=P.pct_change(h).shift(-h); aa=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): aa.append(c)
 aa=np.asarray(aa); print('decay',h,len(aa),round(aa.mean(),6),round(aa.mean()/aa.std(ddof=1),6))
for lab,cut in [('2027+','2027-01-01'),('2028+','2028-01-01'),('2029+','2029-01-01')]:
 q=np.asarray([v for v,d in zip(vals,dates) if d>=pd.Timestamp(cut)]); print(lab,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
out='scripts/miner_2_20291101_trend_quality_signal.csv';pd.DataFrame(rows).to_csv(out,index=False);print('saved',out,len(rows))
