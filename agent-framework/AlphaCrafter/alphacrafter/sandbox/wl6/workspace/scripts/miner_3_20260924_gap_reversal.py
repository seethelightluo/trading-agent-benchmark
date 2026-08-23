import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; S={}
for a in assets:
 p=f'{base}/{a}.csv'
 if not os.path.exists(p): continue
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 gap=d.open/d.close.shift(1)-1
 # fade the completed opening gap; prediction is next close return
 S[a]=pd.DataFrame({'f':-gap,'c':d.close,'r':d.close.shift(-1)/d.close-1})
for h in [1,5,10]:
 rows=[]
 for a,x in S.items():
  q=x.copy(); q.r=x.c.shift(-h)/x.c-1; q=q.dropna()
  rows += [(dt,a,f,r) for dt,f,r in zip(q.index,q.f,q.r)]
 d=pd.DataFrame(rows,columns=['date','a','f','r']); vals=[]; ns=[]
 for dt,g in d.groupby('date'):
  if len(g)>=8: vals.append(spearmanr(g.f,g.r).statistic); ns.append(len(g))
 v=np.array(vals); print(h,'dates',len(v),'avg_n',np.mean(ns),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
# signal validity and turnover among daily ranks
print('assets',len(S),'raw coverage',sum(len(x.dropna()) for x in S.values())/(len(S)*max(len(x) for x in S.values())))
