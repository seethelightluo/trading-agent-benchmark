import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-01-04')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date') for s in U}
# Fade 10d moves only after a pronounced short-volatility shock; neutral otherwise.
for d in px.values():
 r=d.close.pct_change(); d['r10']=d.close.pct_change(10); d['shock']=r.rolling(5).std()/(r.rolling(20).std()+1e-12); d['f']=(-d.r10).where(d.shock>1.25,0.0)
rows=[]
for s,d in px.items():
 for i,dt in enumerate(d.index):
  if i+10>=len(d): continue
  if np.isfinite(d.f.iloc[i]): rows.append((dt,s,d.f.iloc[i],d.close.iloc[i+10]/d.close.iloc[i]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
def stat(z):
 q=[g.factor.corr(g.fwd,method='spearman') for _,g in z.groupby('date') if len(g)>=8 and g.factor.nunique()>1];q=pd.Series(q).dropna();return len(q),z.groupby('date').size().mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
print('shock-gated reversal threshold=1.25; range',x.date.min().date(),x.date.max().date(),'rows',len(x),'assets',x.symbol.nunique())
for label,z in [('all',x),('2020-22',x[x.date<'2023']),('2023-26',x[(x.date>='2023')&(x.date<'2027')]),('2027-29',x[(x.date>='2027')&(x.date<'2030')]),('2030-34',x[x.date>='2030'])]: print(label,stat(z))
for h in [5,10,20,40]:
 q=[]
 for dt,g in x.groupby('date'):
  vals=[]
  for _,rr in g.iterrows():
   d=px[rr.symbol];i=d.index.get_loc(rr.date)
   if i+h<len(d):vals.append((rr.factor,d.close.iloc[i+h]/d.close.iloc[i]-1))
  if len(vals)>=8 and len(set(v[0] for v in vals))>1:q.append(pd.Series([v[0] for v in vals]).corr(pd.Series([v[1] for v in vals]),method='spearman'))
 q=pd.Series(q).dropna();print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('coverage',x.symbol.nunique()/15,'avg_n',x.groupby('date').size().mean(),'turnover',r.diff().abs().mean(axis=1).mean())
