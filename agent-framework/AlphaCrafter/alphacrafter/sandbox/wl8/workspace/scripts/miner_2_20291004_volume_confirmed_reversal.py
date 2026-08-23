import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-10-02']
V=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['volume'] for a in A},axis=1).reindex(P.index)
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); ret5=r.rolling(5,min_periods=5).sum()
# Contrarian short-term move, strengthened by unusually high participation, all lagged.
vs=V/(V.rolling(60,min_periods=30).median())
particip=np.clip(np.log(vs),-1,1)
factor=(-ret5/vol.replace(0,np.nan)*(1+0.35*particip)).shift(1)
fw=P.shift(-10)/P-1
rows=[]
for d in P.index:
 z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.mean()/15,4))
for nm,q in [('full',slice(None)),('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028-29',slice('2028','2029')),('recent360',slice('2028-10-07','2029-10-02')),('recent180',slice('2029-04-07','2029-10-02'))]:
 x=R.loc[q,'ic']; print(nm,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1; vals=[]
 for d in P.index:
  z=pd.concat([factor.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20291004_volume_confirmed_reversal_signal.csv',index=False)
