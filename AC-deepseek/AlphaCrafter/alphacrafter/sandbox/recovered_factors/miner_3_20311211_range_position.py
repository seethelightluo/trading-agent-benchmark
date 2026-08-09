import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index().astype(float)
# range-position persistence: current close relative to rolling 40d low/high, smoothed by 5d median
lo=px.rolling(40,min_periods=40).min(); hi=px.rolling(40,min_periods=40).max()
F=((px-lo)/(hi-lo)).rolling(5,min_periods=5).mean()
R=px.pct_change()
print('DATA',px.index.min().date(),px.index.max().date(),'dates',len(px),'assets',len(A),'coverage',F.notna().mean().mean())
for h in [1,5,10,20]:
 q=[]; ns=[]; turns=[]
 for i in range(40,len(px)-h):
  z=pd.concat([F.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  if i>=10: turns.append(np.mean(F.rank(pct=True).iloc[i].values!=F.rank(pct=True).iloc[i-10].values))
 q=np.array(q); print('H',h,'dates',len(q),'mean_n',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)))
# 10d horizon regimes and recent
for label,mask in [('2020-23',(px.index.year<=2023)),('2024-27',(px.index.year>=2024)&(px.index.year<=2027)),('2028-30',(px.index.year>=2028)&(px.index.year<=2030)),('2031',px.index.year==2031),('recent120',np.arange(len(px))>=len(px)-140)]:
 q=[]
 for i in range(40,len(px)-10):
  if not mask[i]: continue
  z=pd.concat([F.iloc[i],px.iloc[i+10]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('REG',label,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)) if len(q)>1 else 'NA')
print('TURN10',np.mean([np.mean(F.rank(pct=True).iloc[i].values!=F.rank(pct=True).iloc[i-10].values) for i in range(50,len(px))]))
F.to_csv('/tmp/range_position_signal.csv')
