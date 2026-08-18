import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
  D[s]=x.loc[x.index<=pd.Timestamp('2028-02-09')]
 except Exception as e: print('missing',s,e)
rows=[]
for s,x in D.items():
 f=-(x.close/x.close.shift(3)-1)
 for i,dt in enumerate(x.index):
  if pd.notna(f.iloc[i]) and i+1<len(x): rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
a=pd.DataFrame(rows,columns=['date','s','f','y']); ic=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ic.append(spearmanr(g.f,g.y).statistic)
ic=np.asarray(ic,float)
print('idea=3d reversal; dates=%d instruments=%d mean_names=%.2f coverage=%.3f'%(len(ic),len(D),a.groupby('date').size().mean(),a.s.nunique()/15))
print('IC=%.6f ICIR=%.6f hit=%.4f std=%.6f'%(np.mean(ic),np.mean(ic)/(np.std(ic,ddof=1)+1e-12),np.mean(ic>0),np.std(ic,ddof=1)))
for k in [5,10,20]:
 z=[]
 for s,x in D.items():
  f=-(x.close/x.close.shift(3)-1)
  for i,dt in enumerate(x.index):
   if pd.notna(f.iloc[i]) and i+k<len(x): z.append((dt,float(f.iloc[i]),float(x.close.iloc[i+k]/x.close.iloc[i]-1)))
 q=pd.DataFrame(z,columns=['date','f','y']); out=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: out.append(spearmanr(g.f,g.y).statistic)
 out=np.asarray(out); print('%dd IC=%.6f ICIR=%.6f dates=%d'%(k,np.mean(out),np.mean(out)/(np.std(out,ddof=1)+1e-12),len(out)))
print('turnover_proxy=%.6f'%(np.mean([abs(a.f.iloc[i]-a.f.iloc[i-1]) for i in range(1,len(a))])))
