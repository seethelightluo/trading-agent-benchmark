import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 try: return pd.read_csv(p,parse_dates=['date']).query('date<=@cut').set_index('date').close.astype(float)
 except: return pd.read_csv('../persistent/index_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close.astype(float)
p={s:load(s) for s in U}; dxy=load('DXY'); dr=dxy.pct_change(5)
# Dollar-conditioned reversal: reverse recent asset move, with sign flipped in dollar-up/down regimes.
r={s:x.pct_change(5) for s,x in p.items()}; f={s:-r[s]*np.sign(dr.reindex(r[s].index).ffill()) for s in U}
for h in [1,5,10]:
 vals=[]; ns=[]
 for d in sorted(set().union(*[x.index for x in p.values()])):
  a=[]; y=[]
  for s in U:
   if d not in p[s].index or pd.isna(f[s].get(d,np.nan)): continue
   ix=p[s].index.get_loc(d)
   if ix+h>=len(p[s]): continue
   a.append(f[s].loc[d]); y.append(p[s].iloc[ix+h]/p[s].iloc[ix]-1)
  if len(a)>=8 and len(set(a))>1: vals.append(spearmanr(a,y).statistic); ns.append(len(a))
 a=np.array(vals); print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4))
print('regimes')
# daily regime means
rows=[]
for s in U:
 for d in f[s].index:
  if d not in p[s].index: continue
  ix=p[s].index.get_loc(d)
  if ix+1<len(p[s]) and pd.notna(f[s].loc[d]): rows.append((d,s,f[s].loc[d],p[s].iloc[ix+1]/p[s].iloc[ix]-1))
x=pd.DataFrame(rows,columns=['date','s','f','y']); x['yr']=x.date.dt.year
print(x.groupby('date').filter(lambda z:len(z)>=8).groupby('yr').apply(lambda z:z.groupby('date').apply(lambda q:q.f.corr(q.y)).mean()).round(5).to_dict())
print('names',x.s.nunique(),'date_range',x.date.min(),x.date.max())
