import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); r=P.pct_change()
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill().pct_change(); m20=macro.rolling(20).sum(); mr20=r.rolling(20).sum()
vm=macro.rolling(60).mean(); vv=macro.rolling(60).var(); beta=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U: beta[s]=r[s].rolling(60).cov(macro)/vv
sig=(mr20-beta.mul(m20,axis=0)); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill(); reg=(vix<=vix.rolling(120,min_periods=60).median()).astype(float); sig=sig.mul(reg,axis=0).shift(1)
fwd=P.shift(-10).div(P)-1; rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('factor=macro_residual_momentum20_quiet dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/15); print('IC %.8f ICIR %.8f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean()))
for n in [120,252,756,1260]:
 y=x.tail(n); print('recent',n,'IC',y.ic.mean(),'ICIR',y.ic.mean()/y.ic.std(ddof=1))
for h in [1,5,10,20]:
 fw=P.shift(-h).div(P)-1; rr=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',np.mean(rr),'ICIR',np.mean(rr)/np.std(rr,ddof=1),'n',len(rr))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()); sig.to_csv('scripts/miner_1_20340721_macro_residual_momentum20_quiet_signal.csv'); x.to_csv('scripts/miner_1_20340721_macro_residual_momentum20_quiet_ic.csv')
