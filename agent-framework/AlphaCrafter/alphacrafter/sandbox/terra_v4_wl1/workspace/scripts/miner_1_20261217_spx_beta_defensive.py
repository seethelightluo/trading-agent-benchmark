import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]; R=P.pct_change(); m=R['SPX']; wm=m.rolling(60,min_periods=40).mean(); wv=((m-wm)**2).rolling(60,min_periods=40).mean()
rm=R.rolling(60,min_periods=40).mean(); cov=((R-rm).mul(m-wm,axis=0)).rolling(60,min_periods=40).mean(); beta=cov.div(wv,axis=0); f=-beta
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,len(g),round(g.mean(),6),round(g.mean()/g.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'avgvalid',round(f.notna().sum(axis=1).mean(),2))
for n,x in [('rev5',-R.rolling(5).sum()),('mom20',R.rolling(20).sum()),('clv',R)]:
 z=pd.concat([f.stack().rename('f'),x.stack().rename(n)],axis=1).dropna(); print('corr',n,round(z.f.corr(z[n]),5))
