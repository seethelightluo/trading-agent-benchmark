import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2034-02-01')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date') for s in U}
cl=pd.DataFrame({s:d.close for s,d in px.items()}); r=cl.pct_change(); vol=r.rolling(20).std(); f=-r.div(vol+1e-12)
rows=[]
for s,d in px.items():
 for i,dt in enumerate(d.index):
  if i+10<len(d) and np.isfinite(f.loc[dt,s]): rows.append((dt,s,float(f.loc[dt,s]),d.close.iloc[i+10]/d.close.iloc[i]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
q=pd.Series([g.factor.corr(g.fwd,method='spearman') for _,g in x.groupby('date') if len(g)>=8]).dropna()
print('range',x.date.min().date(),x.date.max().date(),'dates',len(q),'assets',x.symbol.nunique(),'avg_n',x.groupby('date').size().mean());print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2034')]:
 z=x[(x.date>=a)&(x.date<=b)];qq=pd.Series([g.factor.corr(g.fwd,method='spearman') for _,g in z.groupby('date') if len(g)>=8]).dropna();print(a,b,len(qq),qq.mean(),qq.mean()/qq.std(ddof=1) if len(qq)>1 else np.nan)
for h in [5,10,20,40]:
 qq=[]
 for dt,g in x.groupby('date'):
  v=[]
  for _,rr in g.iterrows():
   d=px[rr.symbol];i=d.index.get_loc(rr.date)
   if i+h<len(d):v.append((rr.factor,d.close.iloc[i+h]/d.close.iloc[i]-1))
  if len(v)>=8:qq.append(pd.Series([a for a,b in v]).corr(pd.Series([b for a,b in v]),method='spearman'))
 qq=pd.Series(qq).dropna();print('decay',h,qq.mean(),qq.mean()/qq.std(ddof=1))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',r.diff().abs().mean(axis=1).mean(),'coverage',1.0)
