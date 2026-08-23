import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-10-26')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date') for s in U}
for d in px.values():
 d['ret']=d.close.pct_change(); d['r10']=d.close.pct_change(10); d['vol20']=d.ret.rolling(20).std()*np.sqrt(252)
# breadth is contemporaneous, observable at signal date, and smoothed to avoid one-day noise
allret=pd.concat({s:d.ret for s,d in px.items()},axis=1)
breadth=(allret>0).mean(axis=1).rolling(5).mean()
rows=[]
for s,d in px.items():
 for i,dt in enumerate(d.index):
  if i+10>=len(d) or dt not in breadth.index: continue
  v=d.vol20.iloc[i]; br=breadth.loc[dt]
  # mean reversion is strongest when cross-asset breadth is extreme; retain sign and bounded regime multiplier
  mult=1.0+0.6*abs(br-0.5)*2
  f=-d.r10.iloc[i]/v*mult if np.isfinite(v) and v>1e-8 and np.isfinite(br) else np.nan
  fw=d.close.iloc[i+10]/d.close.iloc[i]-1
  if np.isfinite(f) and np.isfinite(fw): rows.append((dt,s,f,fw))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
def stats(z):
 q=[g.factor.corr(g.fwd,method='spearman') for _,g in z.groupby('date') if len(g)>=8];q=pd.Series(q).dropna();return len(q),round(z.groupby('date').size().mean(),2),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),z.symbol.nunique()
print('range',x.date.min(),x.date.max(),'rows',len(x),'assets',x.symbol.nunique());print('overall',stats(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]: print('regime',a,b,stats(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 q=[]
 for dt,g in x.groupby('date'):
  vals=[]
  for _,r in g.iterrows():
   d=px[r.symbol];i=d.index.get_loc(r.date)
   if i+h<len(d): vals.append((r.factor,d.close.iloc[i+h]/d.close.iloc[i]-1))
  if len(vals)>=8:q.append(pd.Series([v[0] for v in vals]).corr(pd.Series([v[1] for v in vals]),method='spearman'))
 q=pd.Series(q).dropna();print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',r.diff().abs().mean(axis=1).mean(),'coverage',x.symbol.nunique()/15,'avg_n',x.groupby('date').size().mean())
