import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}
c=pd.DataFrame(C).sort_index(); r=c.pct_change();
# volatility compression breakout: medium trend times inverse relative volatility (20d vol / 60d vol), lagged signal
f=(r.rolling(20,min_periods=15).sum()*(r.rolling(20,min_periods=15).std()/r.rolling(60,min_periods=40).std()).pow(-1)).shift(0)
for h in [1,5,10,20]:
 y=c.shift(-h)/c-1; z=[]; n=[]
 for d in c.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic);n.append(ok.sum())
 z=np.array(z); print(h,len(z),round(np.mean(n),2),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round(np.mean(z>0),4))
print('coverage',f.notna().mean().mean(),'cells',f.notna().sum().sum())
