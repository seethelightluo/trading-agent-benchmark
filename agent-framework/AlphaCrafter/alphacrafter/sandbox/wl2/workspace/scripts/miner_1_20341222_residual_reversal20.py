import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# 20-day residual reversal: lagged 20d return, cross-sectional de-mean, volatility normalize
mom=P.pct_change(20)
vol=R.rolling(60).std()*np.sqrt(252)
fac=-(mom.sub(mom.median(axis=1),axis=0)).div(vol)
# forward non-overlapping endpoint return from t close to t+h close
out=[]
for dt in fac.index:
 f=fac.loc[dt]; j=P.index.searchsorted(dt,side='right')
 if j+19>=len(P.index): continue
 # Require endpoint 20 trading observations after decision date
 fr=P.iloc[j+19]/P.loc[dt]-1
 z=pd.concat([f,fr],axis=1).dropna()
 if len(z)>=8:
  out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'mean_n',x.n.mean(),'coverage',len(x)/(len(P)-20))
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean(),fac.rank(axis=1).diff().abs().mean(axis=1).mean()/len(U)))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-12-31')]:
 y=x.loc[a:b]; print(a,b,len(y),y.ic.mean(),y.ic.mean()/y.ic.std() if len(y)>1 else np.nan,(y.ic>0).mean())
# write aligned signal artifact
sig=fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); sig.to_csv('../persistent/miner_1_20341222_residual_reversal20_signal.csv',index=False)
