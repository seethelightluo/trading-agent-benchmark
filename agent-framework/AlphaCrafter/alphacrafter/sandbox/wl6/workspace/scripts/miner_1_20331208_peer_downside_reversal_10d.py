import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-12-07')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date') for s in U}
for d in px.values():
 r=d.close.pct_change(); d['r10']=d.close.pct_change(10); d['dn20']=(r.where(r<0)**2).rolling(20,min_periods=5).mean().pow(.5)
rows=[]
for s,d in px.items():
 for i,dt in enumerate(d.index):
  a=d.iloc[i]
  if i+10<len(d) and np.isfinite(a.r10) and np.isfinite(a.dn20) and a.dn20>0: rows.append((dt,s,a.r10,a.dn20))
x=pd.DataFrame(rows,columns=['date','symbol','r10','dn20']); x['med']=x.groupby('date').r10.transform('median'); x['factor']=-(x.r10-x.med)/(x.dn20+1e-12)
x['fwd']=[px[z.symbol].close.iloc[px[z.symbol].index.get_loc(z.date)+10]/px[z.symbol].close.iloc[px[z.symbol].index.get_loc(z.date)]-1 for _,z in x.iterrows()]
def stat(z):
 q=pd.Series([g.factor.corr(g.fwd,method='spearman') for _,g in z.groupby('date') if len(g)>=8]).dropna();return len(q),round(z.groupby('date').size().mean(),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4),z.symbol.nunique()
print('range',x.date.min(),x.date.max(),'rows',len(x),'assets',x.symbol.nunique());print('overall',stat(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]:print('regime',a,b,stat(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 q=[]
 for dt,g in x.groupby('date'):
  vals=[]
  for _,rr in g.iterrows():
   d=px[rr.symbol];j=d.index.get_loc(rr.date)
   if j+h<len(d):vals.append((rr.factor,d.close.iloc[j+h]/d.close.iloc[j]-1))
  if len(vals)>=8:q.append(pd.Series([v[0] for v in vals]).corr(pd.Series([v[1] for v in vals]),method='spearman'))
 q=pd.Series(q).dropna();print('decay',h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',round(r.diff().abs().mean(axis=1).mean(),6),'coverage',round(x.symbol.nunique()/15,4),'avg_n',round(x.groupby('date').size().mean(),2))
