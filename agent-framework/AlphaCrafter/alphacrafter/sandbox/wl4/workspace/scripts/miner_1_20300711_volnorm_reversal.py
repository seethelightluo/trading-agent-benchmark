import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; end=pd.Timestamp('2030-07-10')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date'); px[s]=d.close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:end]; r=P.pct_change(); vol=r.rolling(20).std()
# lagged 5d reversal risk scaled; high means recent losers, adjusted for noise
f=-P.pct_change(5)/(vol*np.sqrt(5))
for h in [5,10,20]:
 rows=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((P.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 a=pd.DataFrame(rows,columns=['date','n','ic']); rec=a[a.date>=a.date.max()-pd.Timedelta(days=365)]
 def met(q): return q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252/h)
 print(h,len(a),round(a.n.mean(),2),*(round(x,6) for x in met(a)), 'recent',*(round(x,6) for x in met(rec)), 'hit',round((a.ic>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),4),'last',P.index[-1].date())
