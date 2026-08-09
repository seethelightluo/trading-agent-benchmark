import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
O=pd.DataFrame({s:D[s].open for s in U}).reindex(P.index); H=pd.DataFrame({s:D[s].high for s in U}).reindex(P.index); L=pd.DataFrame({s:D[s].low for s in U}).reindex(P.index)
# Fade the completed overnight/open-to-close shock, normalized by prior ATR; robustly blend with close location.
tr=(H-L).combine((H-P.shift(1)).abs(),np.maximum).combine((L-P.shift(1)).abs(),np.maximum)
atr=tr.shift(1).rolling(20,min_periods=15).mean()
shock=-(P-O)/atr
clv=((P-L)-(H-P))/(H-L).replace(0,np.nan)
f=(0.7*shock+0.3*(-clv)).replace([np.inf,-np.inf],np.nan)
rows_all={}
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,'dates',len(g),'IC',round(g.mean(),6),'ICIR',round(g.mean()/g.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
print('valid_assets',f.notna().sum(axis=1).mean())
