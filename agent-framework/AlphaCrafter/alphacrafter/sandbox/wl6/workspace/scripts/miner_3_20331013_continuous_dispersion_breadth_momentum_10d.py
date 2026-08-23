import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2033-10-12')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date') for s in U}
for d in px.values():
 r=d.close.pct_change();d['r10']=d.close.pct_change(10);d['breadth']=((r>0).rolling(10).mean()*2-1);d['downsemi']=np.sqrt((r.clip(upper=0)**2).rolling(20).mean())
rel=pd.concat([d.r10.rename(s) for s,d in px.items()],axis=1);med=rel.median(axis=1);disp=rel.std(axis=1);base=disp.rolling(60,min_periods=30).median();rows=[]
for s,d in px.items():
 for dt in d.index:
  i=d.index.get_loc(dt)
  if i+10>=len(d):continue
  den=d.downsemi.loc[dt]; breadth=d.breadth.loc[dt]; ratio=disp.loc[dt]/base.loc[dt] if pd.notna(base.loc[dt]) and base.loc[dt]>1e-10 else np.nan
  # continuous dispersion participation weight, capped at one
  gate=min(1.0,max(0.0,ratio)) if pd.notna(ratio) else np.nan
  f=(d.r10.loc[dt]-med.loc[dt])*breadth*gate/den if pd.notna(den) and den>1e-8 and pd.notna(gate) else np.nan
  fw=d.close.iloc[i+10]/d.close.iloc[i]-1
  if np.isfinite(f) and np.isfinite(fw):rows.append((dt,s,f,fw))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
def stat(z):
 q=[g.factor.corr(g.fwd,method='spearman') for _,g in z.groupby('date') if len(g)>=8];q=pd.Series(q).dropna();return len(q),round(z.groupby('date').size().mean(),2),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),z.symbol.nunique()
print('range',x.date.min(),x.date.max(),'rows',len(x),'assets',x.symbol.nunique());print('overall',stat(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]:print('regime',a,b,stat(x[(x.date>=a)&(x.date<=b)]))
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
