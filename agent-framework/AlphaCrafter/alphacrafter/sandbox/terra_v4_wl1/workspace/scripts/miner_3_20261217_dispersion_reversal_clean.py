import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
R=P.pct_change(); r3=R.rolling(3,min_periods=3).sum(); disp=R.std(axis=1); disp[R.count(axis=1)<8]=np.nan; mu=disp.shift(1).rolling(60,min_periods=30).mean(); sd=disp.shift(1).rolling(60,min_periods=30).std(); z=((disp-mu)/sd).clip(-1,1); amp=(1+0.35*z).clip(.65,1.35); F=-r3.mul(amp,axis=0); F.to_csv('scripts/miner_3_20261217_dispersion_reversal_signal.csv')
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; rows=[]
 for dt in P.index:
  q=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((dt,q.f.corr(q.y,method='spearman'),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year):print('YR',yr,len(g),g.mean())
print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
