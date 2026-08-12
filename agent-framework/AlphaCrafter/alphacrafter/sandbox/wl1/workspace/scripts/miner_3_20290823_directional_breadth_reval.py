import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-08-22')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}; idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
b=sum(w*(r.rolling(h,min_periods=max(3,h//2)).mean()>0).astype(float) for h,w in [(5,.35),(10,.30),(20,.20),(40,.15)]); f=b.sub(b.mean(axis=1),axis=0).shift(1)
print('directional_breadth_revalidation',len(px),px.index.max().date())
for h in [5,10,20]:
 I=[];D=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);D.append(px.index[i])
 a=np.array(I); print(h,len(a),round(a.mean(),6),round(a.mean()/(a.std(ddof=1)+1e-12),6),round((a>0).mean(),4))
 d=pd.DatetimeIndex(D)
 for z in [d>=pd.Timestamp('2029-01-01'),d>=pd.Timestamp('2028-01-01')]:
  x=a[z];print(' recent',len(x),round(x.mean(),6),round(x.mean()/(x.std(ddof=1)+1e-12),6))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean()); f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20290823_directional_breadth_reval_signal.csv',index=False)
