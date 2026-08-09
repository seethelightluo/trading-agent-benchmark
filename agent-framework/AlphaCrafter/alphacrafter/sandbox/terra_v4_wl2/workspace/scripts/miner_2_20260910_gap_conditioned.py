import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:'2026-09-09']
 except Exception as e: print('missing',s)
def evaluate(name, make):
 rows=[]
 for s,x in D.items():
  f=make(x)
  y=x.close.shift(-1)/x.close-1
  q=pd.DataFrame({'f':f,'y':y}).dropna()
  for dt,r in q.iterrows(): rows.append((dt,s,r.f,r.y))
 a=pd.DataFrame(rows,columns=['date','s','f','y']); vals=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append(spearmanr(g.f,g.y).statistic)
 z=np.array(vals); print(name,'dates',len(z),'names',round(a.groupby('date').size().mean(),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'coverage',round(a.date.nunique()/len(set().union(*[set(x.index) for x in D.values()])),4))
for w in [1,3,5,10]:
 evaluate('gap_raw_'+str(w),lambda x,w=w: -(x.open/x.close.shift(1)-1).rolling(w,min_periods=w).mean())
 evaluate('gap_volnorm_'+str(w),lambda x,w=w: -((x.open/x.close.shift(1)-1)/x.close.pct_change().rolling(20,min_periods=15).std()).rolling(w,min_periods=w).mean())
 evaluate('gap_atrnorm_'+str(w),lambda x,w=w: -((x.open/x.close.shift(1)-1)/( (x.high-x.low)/x.close).rolling(20,min_periods=15).mean()).rolling(w,min_periods=w).mean())
# interaction: gap reversal only when prior close location indicates stress
for w in [1,3,5]:
 evaluate('gap_x_range_'+str(w),lambda x,w=w: -((x.open/x.close.shift(1)-1)*((x.high-x.low)/x.close).rolling(10,min_periods=8).mean()).rolling(w,min_periods=w).mean())
