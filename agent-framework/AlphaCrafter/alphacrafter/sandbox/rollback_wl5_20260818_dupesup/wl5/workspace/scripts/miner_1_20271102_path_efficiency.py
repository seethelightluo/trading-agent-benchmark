import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-11-02'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 try: D[s]=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
 except Exception as e: print('missing',s,e)
# Signed path efficiency: net 20d move divided by total absolute daily movement.
def factor(x):
 r=x.close.pct_change()
 return x.close.pct_change(20)/(r.abs().rolling(20).sum()+1e-12)
rec=[]
for s,x in D.items():
 f=factor(x)
 for i,dt in enumerate(x.index[:-1]):
  if dt>END: continue
  if pd.notna(f.iloc[i]) and pd.notna(x.close.iloc[i+1]): rec.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
a=pd.DataFrame(rec,columns=['date','symbol','factor','fwd']); ics=[]
for d,g in a.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: ics.append(spearmanr(g.factor,g.fwd).statistic)
z=np.asarray(ics); print('dates',len(z),'instruments',len(D),'avg_cross_section',a.groupby('date').size().mean(),'coverage',a.symbol.nunique()/15)
print('IC %.6f ICIR %.6f hit %.4f std %.6f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0),np.nanstd(z,ddof=1)))
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2027-11-02')]:
 q=a[(a.date>=lo)&(a.date<=hi)]; zz=[]
 for d,g in q.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:zz.append(spearmanr(g.factor,g.fwd).statistic)
 zz=np.asarray(zz); print(lo,hi,'n',len(zz),'IC %.6f ICIR %.6f'%(np.nanmean(zz),np.nanmean(zz)/np.nanstd(zz,ddof=1)))
for h in [1,5,10]:
 rr=[]
 for s,x in D.items():
  f=factor(x)
  y=x.close.shift(-h)/x.close-1
  q=pd.DataFrame({'f':f,'y':y}).dropna()
  for d,g in q[q.index<=END].groupby(q.index): pass
 # pooled date IC
 for d in sorted(set.intersection(*[set(D[s].index) for s in D])):
  if d>END: continue
  vals=[]
  for s,x in D.items():
   if d in x.index:
    i=x.index.get_loc(d); f=factor(x).iloc[i]; y=x.close.shift(-h).iloc[i]/x.close.iloc[i]-1
    if pd.notna(f) and pd.notna(y): vals.append((f,y))
  if len(vals)>=8: rr.append(spearmanr(np.array(vals)[:,0],np.array(vals)[:,1]).statistic)
 print('horizon',h,'IC',np.nanmean(rr),'n',len(rr))
