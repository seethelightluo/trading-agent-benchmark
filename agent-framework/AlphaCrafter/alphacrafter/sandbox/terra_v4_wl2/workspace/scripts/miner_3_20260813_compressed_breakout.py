import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:'2026-07-15']
 except:pass
# Volatility compression breakout: 5d momentum divided by 20d vol, then penalize currently high vol via 5/20 vol ratio
rows=[]
for s,x in D.items():
 r=x.close.pct_change(); v5=r.rolling(5).std(); v20=r.rolling(20).std(); f=(r.rolling(5).sum()/(v20+1e-12))*(1-v5/(v20+1e-12)).clip(-2,2)
 for i,dt in enumerate(x.index):
  if pd.notna(f.iloc[i]) and i+1<len(x):rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
a=pd.DataFrame(rows,columns=['date','s','f','y']);
for k in [1,5,10]:
 if k>1:
  rows=[]
  for s,x in D.items():
   r=x.close.pct_change();v5=r.rolling(5).std();v20=r.rolling(20).std();f=(r.rolling(5).sum()/(v20+1e-12))*(1-v5/(v20+1e-12)).clip(-2,2)
   for i,dt in enumerate(x.index):
    if pd.notna(f.iloc[i]) and i+k<len(x):rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+k]/x.close.iloc[i]-1)))
  a=pd.DataFrame(rows,columns=['date','s','f','y'])
ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:ics.append(spearmanr(g.f,g.y).statistic)
ic=np.array(ics);print('idea=compressed breakout; dates',len(ic),'avg_names',a.groupby('date').size().mean(),'coverage',a.s.nunique()/15,'horizon',k,'IC %.8f ICIR %.8f hit %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),np.mean(ic>0)))
