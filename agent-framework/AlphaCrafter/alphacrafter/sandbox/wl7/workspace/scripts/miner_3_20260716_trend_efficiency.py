import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 try:
  x=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
  D[s]=x
 except: pass
# Trend efficiency: signed cumulative return divided by total absolute daily returns, 20d
# rows at date t use through t; forward uses next observed close per asset
records=[]
for s,x in D.items():
 r=x.close.pct_change()
 cum=x.close/x.close.shift(20)-1
 eff=cum/(r.abs().rolling(20).sum()+1e-12)
 # alternate efficient directional persistence, bounded naturally
 for dt in x.index:
  if pd.notna(eff.get(dt)):
   nxt=x.index[x.index>dt]
   if len(nxt): records.append((dt,s,float(eff.loc[dt]),float(x.close.loc[nxt[0]]/x.close.loc[dt]-1)))
a=pd.DataFrame(records,columns=['date','s','f','y'])
ics=[]; ranks=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
  ics.append(spearmanr(g.f,g.y).statistic)
  ranks.append(g.f.rank(pct=True).values)
ics=np.array(ics)
print('idea=20d signed trend efficiency; dates',len(ics),'avg names',a.groupby('date').size().mean(),'coverage',a.s.nunique()/15,'valid asset files',len(D))
print('daily IC %.6f ICIR %.6f hit %.4f std %.6f'%(np.nanmean(ics),np.nanmean(ics)/(np.nanstd(ics,ddof=1)+1e-12),np.mean(ics>0),np.nanstd(ics,ddof=1)))
# decay by matching future k-th observation
for k in [5,10,20]:
 rec=[]
 for s,x in D.items():
  r=x.close.pct_change(); cum=x.close/x.close.shift(20)-1; eff=cum/(r.abs().rolling(20).sum()+1e-12)
  for i,dt in enumerate(x.index):
   if pd.notna(eff.iloc[i]) and i+k<len(x): rec.append((dt,float(eff.iloc[i]),float(x.close.iloc[i+k]/x.close.iloc[i]-1)))
 z=pd.DataFrame(rec,columns=['date','f','y']); q=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
 q=np.array(q); print('%dd IC %.6f ICIR %.6f dates %d'%(k,np.nanmean(q),np.nanmean(q)/(np.nanstd(q,ddof=1)+1e-12),len(q)))
# library correlations against factor snapshots on common records, rank correlation cross-section pooled
for fn in glob.glob('factors/*.json'):
 if 'ensemble' in fn or '.bak' in fn: continue
 print('LIB',fn)
