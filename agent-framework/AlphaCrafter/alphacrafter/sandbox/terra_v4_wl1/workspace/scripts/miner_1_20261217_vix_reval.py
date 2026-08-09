import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000300.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# deduplicate universe typo defensively
U=list(dict.fromkeys(U)); cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; macro='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:cut]
v5=v.pct_change(5).reindex(P.index).ffill(); r5=P.pct_change(5); sh=v5.clip(-.5,.5)
f=(-r5.mul(1+sh,axis=0)).where(sh>0,-r5*.5)
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print(h,len(ic),round(a.n.mean(),2),round(ic.mean(),6),round(ic.mean()/ic.std(ddof=1),6),round((ic>0).mean(),4))
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print(yr,len(g),round(g.mean(),6),round(g.mean()/g.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
print('high',int((sh>0).sum()),'total',int(sh.notna().sum()))
plain=-r5; res=-(r5-r5.mean(axis=1))
for n,x in [('plain',plain),('residual',res)]:
 z=pd.concat([f.stack().rename('f'),x.stack().rename(n)],axis=1).dropna(); print('corr',n,round(z.f.corr(z[n]),6))
