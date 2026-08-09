import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-16')
P=pd.DataFrame({s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(fill_method=None); vol=r.rolling(20,min_periods=15).std().shift(1)
# Volatility-normalized three-day reversal, lagged so signal is known before next return.
f=-r.rolling(3,min_periods=3).sum().shift(1)/vol
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; out=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8: out.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1: print('years',[(int(y),round(g.mean(),6),round(g.mean()/g.std(ddof=1),6),len(g)) for y,g in ic.groupby(ic.index.year)])
f.to_csv('scripts/miner_2_20261217_volnorm_reversal_signal.csv',index_label='date')
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5),'assets',len(U),'rows',len(P))
