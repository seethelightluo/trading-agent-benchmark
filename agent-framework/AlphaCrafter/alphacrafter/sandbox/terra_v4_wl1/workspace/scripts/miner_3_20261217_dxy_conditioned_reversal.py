import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill()
dreg=(dxy.shift(1)/dxy.shift(61)-1).clip(-.10,.10)
macro=((dreg-dreg.rolling(252,min_periods=60).mean())/dreg.rolling(252,min_periods=60).std()).replace([np.inf,-np.inf],np.nan).clip(-2,2)
rev=-(P.shift(1)/P.shift(2)-1)
def z(x):
 sd=x.std(axis=1).replace(0,np.nan); return x.sub(x.mean(axis=1),axis=0).div(sd,axis=0)
f=z(rev)*(1+.35*macro.fillna(0).abs())
f.to_csv('scripts/miner_3_20261217_dxy_conditioned_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']); a['date']=pd.to_datetime(a.date); a=a.set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1:
  for yr,g in ic.groupby(ic.index.to_series().dt.year): print('YR',yr,'n',len(g),'IC',round(g.mean(),5),'ICIR',round(g.mean()/g.std(ddof=1),4))
print('coverage',round(f.notna().sum().sum()/f.size,5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
print('period',P.index.min().date(),P.index.max().date())
