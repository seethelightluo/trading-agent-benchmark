import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data/'; w=pd.DataFrame({s:pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2030-08-21']; r=w.pct_change(); path=r.abs().rolling(20).sum()
# High score means a choppy, inefficient 20-day decline/reversal opportunity; causal and volatility scaled.
fac=-(r.rolling(20).sum()/path)*(r.rolling(20).std().replace(0,np.nan)**-1)
rows=[]
for i in range(len(w)-10):
 z=pd.concat([fac.iloc[i],w.iloc[i+10]/w.iloc[i]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): rows.append((w.index[i],q,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(x),'mean_n',round(x.n.mean(),3),'coverage',round(x.n.mean()/15,4)); print('IC10',round(x.ic.mean(),5),'ICIR',round(x.ic.mean()/x.ic.std(),5),'hit',round((x.ic>0).mean(),4))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-08-21')]:
 y=x.loc[a:b].ic; print('REG',a[:4],len(y),round(y.mean(),5),round(y.mean()/y.std(),5) if len(y)>1 else np.nan)
print('turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5)); fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20300822_inverse_efficiency_signal.csv',index=False)
