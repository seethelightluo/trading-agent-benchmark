import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2033-09-01')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date'); return d
px={s:load(s) for s in U}
for s,d in px.items():
 r=d.close.pct_change(); d['r10']=d.close.pct_change(10); d['breadth']=((r>0).rolling(10).mean()*2-1); d['downvol']=r.where(r<0).rolling(20).std()*np.sqrt(20)
rel=pd.concat([d.r10.rename(s) for s,d in px.items()],axis=1); med=rel.median(axis=1); rows=[]
for s,d in px.items():
 for dt in d.index.intersection(med.index):
  i=d.index.get_loc(dt); den=d.loc[dt,'downvol']
  if i+10<len(d) and pd.notna(d.loc[dt,'r10']) and pd.notna(den) and den>1e-8:
   f=(d.loc[dt,'r10']-med.loc[dt])*d.loc[dt,'breadth']/den; fw=d.close.iloc[i+10]/d.close.iloc[i]-1
   if np.isfinite(f) and np.isfinite(fw): rows.append((dt,s,f,fw))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
def stat(z):
 q=[]
 for _,g in z.groupby('date'):
  if len(g)>=8:q.append(g.factor.corr(g.fwd,method='spearman'))
 q=pd.Series(q).dropna();return len(q),round(z.groupby('date').size().mean(),2),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),z.symbol.nunique()
print('range',x.date.min(),x.date.max(),'rows',len(x),'assets',x.symbol.nunique());print('overall',stat(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]:print('regime',a,b,stat(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 q=[]
 for dt,g in x.groupby('date'):
  v=[]
  for _,r in g.iterrows():
   d=px[r.symbol];i=d.index.get_loc(r.date)
   if i+h<len(d):v.append((r.factor,d.close.iloc[i+h]/d.close.iloc[i]-1))
  if len(v)>=8:q.append(pd.DataFrame(v,columns=['a','b']).a.corr(pd.DataFrame(v,columns=['a','b']).b,method='spearman'))
 q=pd.Series(q).dropna();print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',r.diff().abs().mean(axis=1).mean(),'coverage',x.symbol.nunique()/15,'avg_n',x.groupby('date').size().mean())
