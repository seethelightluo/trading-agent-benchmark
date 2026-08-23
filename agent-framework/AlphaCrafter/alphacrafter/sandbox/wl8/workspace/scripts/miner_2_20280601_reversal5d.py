import os,pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d['date']=pd.to_datetime(d.date);p[s]=d.sort_values('date').set_index('date').close
px=pd.concat(p,axis=1).sort_index(); r=px.pct_change(); sig=-px.shift(1)/px.shift(6)+1; fw=px.shift(-1)/px-1
rows=[]
for dt in px.index:
 if dt>pd.Timestamp('2028-05-31'):continue
 z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'rows',a.n.sum(),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
for nm,sub in [('all',a),('2020-22',a.loc['2020':'2022']),('2023-25',a.loc['2023':'2025']),('2026',a.loc['2026']),('2027',a.loc['2027']),('2028',a.loc['2028']),('last180',a.tail(180))]:
 if len(sub): print(nm,len(sub),'IC %.6f ICIR %.6f hit %.3f'%(sub.ic.mean(),sub.ic.mean()/(sub.ic.std(ddof=1)+1e-12)*np.sqrt(len(sub)),(sub.ic>0).mean()))
print('turnover',sig.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [3,5,10]:
 fw=px.shift(-h)/px-1; q=[]
 for dt in px.index:
  if dt>pd.Timestamp('2028-05-31'):continue
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'dates',len(q),'IC %.6f ICIR %.6f'%(np.mean(q),np.mean(q)/(np.std(q,ddof=1)+1e-12)*np.sqrt(len(q))))
