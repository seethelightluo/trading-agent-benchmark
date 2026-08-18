import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index()
r=np.log(px).diff(); sig=(-r.rolling(20,min_periods=15).std()).shift(1)
ics={h:[] for h in [5,10,20]}; dates=[]; ns=[]
for dt in sig.index:
 for h in [5,10,20]:
  y=np.log(px.shift(-h)/px); ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: ics[h].append(spearmanr(sig.loc[dt][ok],y.loc[dt][ok]).statistic)
for h in [5,10,20]:
 z=pd.Series(ics[h]).dropna(); print('h',h,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
print('avg coverage',sig.notna().sum(axis=1).mean()/15,'signal coverage',sig.notna().stack().mean())
rk=sig.rank(axis=1,pct=True);print('turnover',(rk-rk.shift()).abs().mean(axis=1).dropna().mean())
# artifact at admission h10
ys=np.log(px.shift(-10)/px); rows=[]
for dt in sig.index:
 for s in U: rows.append((dt,s,sig.loc[dt,s]))
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20331028_inversevol20_signal.csv',index=False)
