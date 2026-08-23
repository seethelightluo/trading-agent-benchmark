import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-02-01')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date') for s in U}
cl=pd.DataFrame({s:d.close for s,d in px.items()}); r=cl.pct_change(); r10=cl.pct_change(10); prior=r10.shift(10); vol=r.rolling(30).std()*np.sqrt(252)
# Acceleration of 10d momentum, normalized by trailing 30d realized volatility.
factor=(r10-prior)/(vol+1e-12); factor=factor.sub(factor.median(axis=1),axis=0)
rows=[]
for s,d in px.items():
 for i,dt in enumerate(d.index):
  if i+10<len(d) and np.isfinite(factor.loc[dt,s]): rows.append((dt,s,float(factor.loc[dt,s]),d.close.iloc[i+10]/d.close.iloc[i]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']);x.date=pd.to_datetime(x.date)
def stat(z):
 q=[g.factor.corr(g.fwd,method='spearman') for _,g in z.groupby('date') if len(g)>=8];q=pd.Series(q).dropna();return len(q),round(z.groupby('date').size().mean(),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4),z.symbol.nunique()
print('factor=relative acceleration (r10-r10_lag10)/annualized vol30');print('range',x.date.min().date(),x.date.max().date(),'rows',len(x),'assets',x.symbol.nunique());print('overall',stat(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2034')]:print('regime',a,b,stat(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 q=[]
 for dt,g in x.groupby('date'):
  vals=[]
  for _,rr in g.iterrows():
   d=px[rr.symbol];i=d.index.get_loc(rr.date)
   if i+h<len(d): vals.append((rr.factor,d.close.iloc[i+h]/d.close.iloc[i]-1))
  if len(vals)>=8:q.append(pd.Series([v[0] for v in vals]).corr(pd.Series([v[1] for v in vals]),method='spearman'))
 q=pd.Series(q).dropna();print('decay',h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
rk=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',round(rk.diff().abs().mean(axis=1).mean(),6),'coverage',round(x.symbol.nunique()/15,4),'avg_n',round(x.groupby('date').size().mean(),2))
