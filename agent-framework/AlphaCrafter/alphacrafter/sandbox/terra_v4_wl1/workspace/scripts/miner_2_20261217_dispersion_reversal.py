import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); disp=r.rolling(20,min_periods=12).std().mean(axis=1).shift(1)
# Reversal is stronger during high cross-sectional dispersion, mild otherwise; lag all inputs.
dispz=(disp/disp.rolling(120,min_periods=40).median()-1).clip(-.7,.7)
f=-r.rolling(5,min_periods=5).sum().shift(1).mul(1+dispz,axis=0)
f=f.sub(f.median(axis=1),axis=0)
Y=P.shift(-1).div(P)-1; rows=[]
for d in P.index:
 q=pd.concat([f.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
 if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
print('dates',len(ic),'avg_n',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for yr,g in ic.groupby(ic.index.year): print('year',yr,len(g),g.mean(),g.mean()/g.std(ddof=1))
for h in [5,10]:
 yy=P.shift(-h).div(P)-1; z=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),yy.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(q.f.corr(q.y))
 z=pd.Series(z).dropna();print('decay',h,z.mean(),z.mean()/z.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20261217_dispersion_reversal_signal.csv',index_label='date');print('ARTIFACT scripts/miner_2_20261217_dispersion_reversal_signal.csv')
