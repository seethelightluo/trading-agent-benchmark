import pandas as pd,numpy as np
from scipy.stats import spearmanr
root='../persistent'; A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for a in A:
 d=pd.read_csv(f'{root}/stock_data/{a}.csv');d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index()
 # range shock: today's intraday range relative to trailing 20d median, signed by close location; fade extreme pressure
 r=(d.high-d.low)/d.close
 cl=2*(d.close-d.low)/(d.high-d.low).replace(0,np.nan)-1
 d['f']=-(cl*(r/r.rolling(20,min_periods=15).median()))
 d['fwd']=d.close.shift(-1)/d.close-1;d['asset']=a;rows.append(d[['f','fwd','asset']].reset_index())
x=pd.concat(rows).dropna(); out=[]
for dt,g in x.groupby('date'):
 if len(g)>=8: out.append((dt,spearmanr(g.f,g.fwd).statistic,len(g)))
o=pd.DataFrame(out,columns=['date','ic','n'])
print('range shock signed fade dates',len(o),'avgN',o.n.mean(),'coverage',len(x)/(15*len(set(x.date))))
print('daily',o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean(),o.ic.std(ddof=1))
for y in range(2020,2027):
 z=o[o.date.dt.year==y]; print(y,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>2 else np.nan)
for h in [5,10]:
 rr=[]
 for a in A:
  d=pd.read_csv(f'{root}/stock_data/{a}.csv');d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index();r=(d.high-d.low)/d.close;cl=2*(d.close-d.low)/(d.high-d.low).replace(0,np.nan)-1;d['f']=-(cl*r/r.rolling(20,min_periods=15).median());d['fwd']=d.close.shift(-h)/d.close-1;d['asset']=a;rr.append(d[['f','fwd','asset']].reset_index())
 z=pd.concat(rr).dropna(); q=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8:q.append(spearmanr(g.f,g.fwd).statistic)
 print(h,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
