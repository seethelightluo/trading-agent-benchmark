import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-12-20'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv');vix.date=pd.to_datetime(vix.date); v=vix.set_index('date').close.astype(float).reindex(P.index).ffill()
# VIX-regime conditioned downside trend: reward persistent positive trend in calm regimes, invert/defensive response in stressed regimes.
vol=R.rolling(40,min_periods=25).std().shift(1); mom=P.pct_change(20).shift(1).div(vol+1e-12)
cons=(R.shift(1).gt(0).rolling(20,min_periods=15).mean()-.5)*2
vz=(v.shift(1)-v.shift(1).rolling(126,min_periods=60).mean())/(v.shift(1).rolling(126,min_periods=60).std()+1e-12)
# smooth regime coefficient, fully lagged
reg=np.tanh(-vz/2)
F=mom*cons*reg.values[:,None]
F=pd.DataFrame(F,index=P.index,columns=P.columns);F.to_csv('scripts/miner_3_20351221_vix_regime_trend_signal.csv')
for h in [1,3,5,10]:
 a=[]; dates=[]
 for i in range(45,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):a.append(q);dates.append(P.index[i])
 a=pd.Series(a,index=dates); print('h',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'recent252',round(a.tail(252).mean(),6),'recentIR',round(a.tail(252).mean()/a.tail(252).std(ddof=1),6))
 print('blocks',*[round(x.mean(),6) for x in np.array_split(a,4)])
print('coverage',round(F.notna().mean().mean(),4),'turn',round(F.rank(axis=1).diff().abs().mean().mean()/15,6))
