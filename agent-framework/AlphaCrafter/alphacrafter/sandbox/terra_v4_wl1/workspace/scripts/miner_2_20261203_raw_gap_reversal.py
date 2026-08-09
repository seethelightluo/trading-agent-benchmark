import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-12-02');b='../persistent/stock_data'
D={s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close.astype(float) for s in U}).sort_index();P=P[P.index<=cut]
O=pd.DataFrame({s:D[s].open.astype(float) for s in U}).reindex(P.index); prev=P.shift(1)
f=-(O/prev-1); Y=P.shift(-1)/P-1
rows=[]
for d in P.index:
 q=pd.concat([f.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
 if len(q)>=8:rows.append((d,q.f.corr(q.y),len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');ic=a.ic
print('dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
print('years',[(int(y),round(g.mean(),5),len(g)) for y,g in ic.groupby(ic.index.year)])
print('coverage',round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for h in [5,10]:
 Y=P.shift(-h)/P-1; rr=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:rr.append(q.f.corr(q.y))
 z=pd.Series(rr);print('H',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
