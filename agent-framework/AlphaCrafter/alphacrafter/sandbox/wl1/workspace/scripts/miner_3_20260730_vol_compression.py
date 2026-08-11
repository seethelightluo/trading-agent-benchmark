import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]
x=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=x.pct_change()
# Volatility compression: current short vol relative to long vol, inverted (compressed assets favored)
for sw,lw in [(5,20),(10,40),(20,60)]:
 sv=r.rolling(sw).std(); lv=r.rolling(lw).std(); f=-(sv/lv)
 for h in [5,10]:
  rows=[]
  for s in U:
   ix=x[s].dropna().index
   for j in range(lw,len(ix)-h):
    dt=ix[j]; z=f.loc[dt,s]; y=x.loc[ix[j+h],s]/x.loc[dt,s]-1
    if pd.notna(z) and pd.notna(y): rows.append((dt,s,z,y))
  a=pd.DataFrame(rows,columns=['date','s','f','y']); q=[]
  for dt,g in a.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
  q=np.array(q); print('sw/lw',sw,lw,'h',h,'dates',len(q),'avgN',round(a.groupby('date').size().mean(),2),'coverage',round(a.groupby('date').size().mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
  print('regime', {int(y):round(pd.Series(q,index=sorted(a.date.unique())[:len(q)])[pd.Series(sorted(a.date.unique())[:len(q)]).dt.year==y].mean(),5) for y in []})
