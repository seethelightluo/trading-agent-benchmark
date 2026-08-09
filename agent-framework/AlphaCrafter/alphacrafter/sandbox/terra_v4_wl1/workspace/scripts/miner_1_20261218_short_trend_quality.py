import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}; P=pd.DataFrame(D).sort_index().loc[:cut]
f=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 x=P[s].dropna(); r=x.pct_change(fill_method=None); mom=x.shift(1).div(x.shift(11))-1; vol=r.rolling(20,min_periods=12).std().shift(1)
 f.loc[x.index,s]=(mom/(vol*np.sqrt(20)+1e-8)).where(mom.notna()&vol.notna())
f.to_csv('scripts/miner_1_20261218_short_trend_quality_signal.csv',index_label='date')
y=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 x=P[s].dropna(); z=x.shift(-1).div(x)-1; y.loc[z.index,s]=z
rows=[]
for d in P.index:
 q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
 if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
print('dates',len(ic),'avg_n',round(a.n.mean(),3),'IC',round(ic.mean(),8),'ICIR',round(ic.mean()/ic.std(ddof=1),8),'hit',round((ic>0).mean(),4))
for yr,g in ic.groupby(ic.index.year): print('YR',yr,'n',len(g),'IC',round(g.mean(),6),'ICIR',round(g.mean()/g.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
