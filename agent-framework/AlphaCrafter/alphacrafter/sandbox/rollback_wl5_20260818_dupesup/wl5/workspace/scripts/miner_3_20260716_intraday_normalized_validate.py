import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2026-07-15'").set_index('date').sort_index() for s in U}
variants={}
for s,x in D.items():
 rng=(x.high-x.low); variants.setdefault(s,{})['norm']=-(x.close-x.open)/rng; variants[s]['body']=-(x.close-x.open)/x.open; variants[s]['clvbody']=-(x.close-x.open)/rng* np.sqrt(rng/x.close)
def test(key):
 rows=[]
 for s,x in D.items():
  f=variants[s][key]; r=x.close.shift(-1)/x.close-1
  rows.append(pd.DataFrame({'date':x.index,'f':f.values,'r':r.values,'s':s}))
 a=pd.concat(rows).dropna(); out=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8:
   c=g.f.corr(g.r,method='spearman')
   if pd.notna(c):out.append((dt,c))
 z=pd.DataFrame(out,columns=['date','ic']).set_index('date'); return z
for k in variants[U[0]]:
 z=test(k); print(k,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean())
# pooled corr with library rough factors
for k in variants[U[0]]:
 ss=[]
 for s,x in D.items():
  rng=x.high-x.low; f=variants[s][k]; clv=-(2*(x.close-x.low)/rng-1); rev=-(x.close/x.close.shift(5)-1); mom=x.close/x.close.shift(20)-1
  ss.append(pd.DataFrame({'f':f,'clv':clv,'rev':rev,'mom':mom}))
 print(k,pd.concat(ss).dropna().corr().f.round(4).to_dict())
