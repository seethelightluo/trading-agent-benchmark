import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2026-07-29'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date <= @cut').set_index('date').sort_index() for s in U}
# Bounded range/body hybrid: intraday reversal weighted by candle range relative to close.
rows=[]
for s,x in D.items():
    rng=(x.high-x.low).replace(0,np.nan)
    f=-(x.close-x.open)/rng*np.sqrt(rng/x.close)
    r=x.close.shift(-1)/x.close-1
    rows.append(pd.DataFrame({'date':x.index,'f':f,'r':r,'symbol':s}).reset_index(drop=True))
a=pd.concat(rows,ignore_index=True).dropna()
out=[]
for dt,g in a.groupby('date'):
    if len(g)>=8:
        ic=g.f.corr(g.r,method='spearman')
        if pd.notna(ic): out.append((dt,ic,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
rank=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank').sort_index()
turn=rank.diff().abs().mean(axis=1).mean()
print('factor clvbody_volscaled; cutoff',cut,'dates',len(z),'avg_n',z.n.mean(),'coverage',len(a)/sum(len(x) for x in D.values()),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'turnover',turn)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-29')]:
 q=z.loc[lo:hi].ic; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [1,5,10]:
 rr=[]
 for s,x in D.items():
  rng=(x.high-x.low).replace(0,np.nan); f=-(x.close-x.open)/rng*np.sqrt(rng/x.close); fr=x.close.shift(-h)/x.close-1
  rr.append(pd.DataFrame({'date':x.index,'f':f,'r':fr}))
 b=pd.concat(rr,ignore_index=True).dropna(); vals=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8:
   c=g.f.corr(g.r,method='spearman')
   if pd.notna(c): vals.append(c)
 print('decay',h,len(vals),np.mean(vals))
# Artifact correlations with established signals, pooled date-symbol observations.
fs=[]
for s,x in D.items():
 rng=(x.high-x.low).replace(0,np.nan)
 fs.append(pd.DataFrame({'candidate':-(x.close-x.open)/rng*np.sqrt(rng/x.close),'clv':-(2*(x.close-x.low)/rng-1),'rev5':-(x.close/x.close.shift(5)-1),'mom20':x.close/x.close.shift(20)-1}))
print('correlations',pd.concat(fs,ignore_index=True).dropna().corr()['candidate'].round(4).to_dict())
