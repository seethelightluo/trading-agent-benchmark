import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-01-04')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date') for s in U}
# Volatility-shock reversal: fade recent 10d move only when short volatility is elevated versus its medium baseline.
for d in px.values():
 r=d.close.pct_change(); d['r10']=d.close.pct_change(10); d['v5']=r.rolling(5).std(); d['v20']=r.rolling(20).std(); d['f']=-d.r10*(d.v5/(d.v20+1e-12)-1.0)
rows=[]
for s,d in px.items():
 for i,dt in enumerate(d.index):
  if i+10>=len(d): continue
  if np.isfinite(d.f.iloc[i]): rows.append((dt,s,d.f.iloc[i],d.close.iloc[i+10]/d.close.iloc[i]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
def stat(z):
 q=[g.factor.corr(g.fwd,method='spearman') for _,g in z.groupby('date') if len(g)>=8]; q=pd.Series(q).dropna(); return {'dates':len(q),'avg_n':round(z.groupby('date').size().mean(),2),'IC':round(q.mean(),6),'ICIR':round(q.mean()/q.std(ddof=1),6),'hit':round((q>0).mean(),4),'assets':z.symbol.nunique()}
print('factor=negative 10d return times (5d realized vol / 20d realized vol - 1)')
print('range',x.date.min().date(),x.date.max().date(),'rows',len(x),'assets',x.symbol.nunique());print('overall',stat(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2034')]: print('regime',a,b,stat(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 q=[]
 for dt,g in x.groupby('date'):
  vals=[]
  for _,rr in g.iterrows():
   d=px[rr.symbol]; i=d.index.get_loc(rr.date)
   if i+h<len(d): vals.append((rr.factor,d.close.iloc[i+h]/d.close.iloc[i]-1))
  if len(vals)>=8:q.append(pd.Series([z[0] for z in vals]).corr(pd.Series([z[1] for z in vals]),method='spearman'))
 q=pd.Series(q).dropna();print('decay',h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('turnover',round(r.diff().abs().mean(axis=1).mean(),6),'coverage',round(x.symbol.nunique()/15,4),'avg_n',round(x.groupby('date').size().mean(),2))
