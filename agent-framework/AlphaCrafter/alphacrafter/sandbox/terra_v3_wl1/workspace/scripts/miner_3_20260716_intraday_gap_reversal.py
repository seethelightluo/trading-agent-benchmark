import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 D[s]=x
# prior completed bar intraday return, factor is reversal of intraday move
rows=[]
for s,x in D.items():
 z=pd.DataFrame({'f':-(x.close/x.open-1), 'close':x.close})
 z['fr']=z.close.shift(-1)/z.close-1
 z['s']=s; rows.append(z.reset_index())
a=pd.concat(rows).dropna(subset=['f','fr'])
# only dates with >=8
out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8: out.append((dt,g.f.corr(g.fr,method='spearman'),len(g)))
r=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'mean n',r.n.mean(),'coverage',len(r)/len(pd.date_range('2020-01-01','2026-07-15')))
print('daily IC %.5f ICIR %.5f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean()))
for h in [5,10,20]:
 rr=[]
 for s,x in D.items():
  z=pd.DataFrame({'f':-(x.close/x.open-1),'fr':x.close.shift(-h)/x.close-1}).dropna();z['date']=z.index; z=z.reset_index(drop=True); rr.append(z)
 q=pd.concat(rr)
 vals=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8: vals.append(g.f.corr(g.fr,method='spearman'))
 vals=pd.Series(vals).dropna();print(h,'d',len(vals),'IC',vals.mean(),'ICIR',vals.mean()/vals.std())
# corr against approximated existing factors
base=[]
for s,x in D.items():
 z=pd.DataFrame({'cand':-(x.close/x.open-1),'rev':-(x.close/x.close.shift(5)-1),'mom':x.close/x.close.shift(20)-1,'clv':((x.close-x.low)-(x.high-x.close))/(x.high-x.low).replace(0,np.nan)})
 z['s']=s;z['date']=z.index;base.append(z)
b=pd.concat(base).dropna()
print('corr',b[['cand','rev','mom','clv']].corr().cand.to_dict())
