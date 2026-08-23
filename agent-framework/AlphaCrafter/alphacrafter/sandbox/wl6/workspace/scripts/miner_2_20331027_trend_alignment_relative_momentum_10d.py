import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2033-10-26')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date') for s in U}
for d in px.values():
 r=d.close.pct_change(); d['r10']=d.close.pct_change(10); d['r20']=d.close.pct_change(20); d['ds']=np.sqrt((r.clip(upper=0)**2).rolling(20).mean()); d['alignment']=((r>0).rolling(10).mean()*2-1)
rel=pd.concat([d.r10.rename(s) for s,d in px.items()],axis=1); med=rel.median(axis=1); rows=[]
for s,d in px.items():
 for dt in d.index:
  i=d.index.get_loc(dt)
  if i+10>=len(d): continue
  ds=d.ds.loc[dt]; a=d.alignment.loc[dt]; r20=d.r20.loc[dt]; f=((d.r10.loc[dt]-med.loc[dt])*(1+0.5*a)*np.sign(r20) if pd.notna(r20) else np.nan)
  f=f/ds if pd.notna(ds) and ds>1e-8 else np.nan
  fw=d.close.iloc[i+10]/d.close.iloc[i]-1
  if np.isfinite(f) and np.isfinite(fw): rows.append((dt,s,f,fw))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
def stat(z):
 q=[g.factor.corr(g.fwd,method='spearman') for _,g in z.groupby('date') if len(g)>=8];q=pd.Series(q).dropna();return len(q),round(z.groupby('date').size().mean(),2),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),z.symbol.nunique()
print('range',x.date.min(),x.date.max(),'rows',len(x),'assets',x.symbol.nunique()); print('overall',stat(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]: print('regime',a,b,stat(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 q=[]
 for dt,g in x.groupby('date'):
  vals=[]
  for _,rr in g.iterrows():
   d=px[rr.symbol];i=d.index.get_loc(rr.date)
   if i+h<len(d): vals.append((rr.factor,d.close.iloc[i+h]/d.close.iloc[i]-1))
  if len(vals)>=8:q.append(pd.Series([v[0] for v in vals]).corr(pd.Series([v[1] for v in vals]),method='spearman'))
 q=pd.Series(q).dropna(); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean(),'coverage',x.symbol.nunique()/15,'avg_n',x.groupby('date').size().mean())
