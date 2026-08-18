import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
def run(kind):
 rows=[]
 for s,x in D.items():
  prev=x.close.shift(1)
  f=-(x.open/prev-1) if kind=='gap' else -(x.close/x.open-1)
  z=pd.DataFrame({'f':f,'c':x.close,'s':s});z['fr']=z.c.shift(-1)/z.c-1;rows.append(z.reset_index())
 a=pd.concat(rows).dropna(); vals=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8: vals.append(g.f.corr(g.fr,method='spearman'))
 v=pd.Series(vals).dropna();print(kind,len(v),v.mean(),v.mean()/v.std(),(v>0).mean())
for k in ['gap','intra']:run(k)
