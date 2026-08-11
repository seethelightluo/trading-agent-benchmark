import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.rename(s)
p=pd.concat([L(s) for s in A],axis=1)
# medium horizon risk-adjusted trend: 60d return divided by recent 30d annualized vol
f=(p/p.shift(60)-1).div(p.pct_change().rolling(30).std()*np.sqrt(252))
for h in [5,10,20]:
 out=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: out.append((p.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
 print(h,len(q),q.n.mean(),q.n.mean()/15,ic,ir,(q.ic>0).mean())
 for lab,st in [('2026','2026-01-01'),('2027','2027-01-01'),('2028','2028-01-01')]:
  x=q[q.index>=st]; print(lab,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_1_20280420_medium_risk_trend_signal.csv',index_label='date')
