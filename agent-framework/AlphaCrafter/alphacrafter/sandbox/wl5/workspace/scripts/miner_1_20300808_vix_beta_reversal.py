import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data/'; wide=pd.DataFrame({s:pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); wide=wide.loc[:'2030-08-07']; ret=wide.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(wide.index).ffill(); vr=v.pct_change()
base=(-wide.pct_change(5).clip(upper=0))/ret.rolling(20).std()*np.sqrt((ret.rolling(60).std()/ret.rolling(20).std()).clip(.25,4))
beta=ret.rolling(60).cov(vr).div(vr.rolling(60).var(),axis=0).clip(-3,3)
g=(1+beta*vr.rolling(5).sum().clip(-.5,.5).values[:,None]).clip(.25,1.75)
fac=base*g
rows=[]
for i in range(len(wide)-10):
 z=pd.concat([fac.iloc[i],wide.iloc[i+10]/wide.iloc[i]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): rows.append((wide.index[i],q,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(x),'mean_n',x.n.mean(),'coverage',x.n.mean()/15); print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(),'hit',(x.ic>0).mean());
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-08-07')]:
 y=x.loc[a:b].ic; print(a[:4],len(y),y.mean(),y.mean()/y.std())
r=fac.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean()); out=fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20300808_vix_beta_reversal_signal.csv',index=False)
