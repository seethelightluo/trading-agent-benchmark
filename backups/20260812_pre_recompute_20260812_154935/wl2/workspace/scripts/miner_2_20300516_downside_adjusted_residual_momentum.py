import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in symbols:
 try: d=get_stock_daily_data(s,days=4000)
 except Exception: d=None
 if d is None:
  try: d=get_index_daily_data(s,days=4000)
  except Exception: d=None
 if d is not None and len(d)>100:
  x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); frames[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(frames).sort_index(); r=p.pct_change(); mom=p.pct_change(20); down=r.clip(upper=0).rolling(20).std(); raw=mom/(down*np.sqrt(20)+1e-8); fac=raw.sub(raw.median(axis=1),axis=0)
for h in [1,5,10]:
 fwd=p.shift(-h)/p-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna(); print('H',h,'n',len(a),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('dates',len(p),'instruments',len(frames),'coverage',fac.notna().mean().mean(),'avg cross',fac.notna().sum(axis=1).mean())
for start in ['2027-01-01','2028-01-01','2029-01-01','2029-07-01','2030-01-01']:
 fwd=p.shift(-1)/p-1; q=[]
 for dt in fac.loc[start:].index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(start,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20300516_downside_adjusted_residual_momentum_signal.csv',index=False)
