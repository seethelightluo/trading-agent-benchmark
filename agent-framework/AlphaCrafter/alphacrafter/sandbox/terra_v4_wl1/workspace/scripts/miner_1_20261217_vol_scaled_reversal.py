import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); f=-P.pct_change(5).div(vol,axis=0)
rows=[]
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; out=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: out.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); z=a.ic
 print('H',h,'dates',len(z),'avg_n',round(a.n.mean(),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
 if h==1:
  for yr,g in z.groupby(z.index.year): print('YR',yr,len(g),round(g.mean(),6),round(g.mean()/g.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
print('corr raw5',pd.concat([f.stack(),(-P.pct_change(5)).stack()],axis=1).dropna().corr().iloc[0,1])
# save reproducible artifact
f.to_csv('scripts/miner_1_20261217_vol_scaled_reversal_signal.csv')
