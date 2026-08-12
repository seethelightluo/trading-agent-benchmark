import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-09-05')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}; idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
b=sum(w*(r.rolling(h,min_periods=max(3,h//2)).mean()>0).astype(float) for h,w in [(5,.35),(10,.30),(20,.20),(40,.15)]); f=b.sub(b.mean(axis=1),axis=0).shift(1)
print('directional breadth current',len(px),px.index.max().date())
for h in [5,10,20]:
 I=[];D=[];N=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);D.append(px.index[i]);N.append(len(q))
 a=np.array(I); d=pd.DatetimeIndex(D);print('H',h,'dates',len(a),'N',np.mean(N),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 for lab,m in [('2028+',d>=pd.Timestamp('2028-01-01')),('2029',d>=pd.Timestamp('2029-01-01'))]:
  x=a[m];print(lab,len(x),x.mean(),x.mean()/x.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20290906_directional_breadth_signal.csv',index=False)
