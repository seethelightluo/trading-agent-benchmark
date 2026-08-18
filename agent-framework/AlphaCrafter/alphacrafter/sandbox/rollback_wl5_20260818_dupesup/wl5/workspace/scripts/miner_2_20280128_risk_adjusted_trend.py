import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-01-28')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=end].sort_values('date').set_index('date'); D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change();
# low volatility with medium-term trend: positive trend divided by realized vol, an interpretable risk-adjusted trend
mom=p.pct_change(20); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252); f=mom/(vol+1e-8)
for h in [1,5,10,20]:
 q=p.shift(-h)/p-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('h',h,'dates',len(a),'mean_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
 for label,x in [('2020-22',a.loc['2020':'2022']),('2023-25',a.loc['2023':'2025']),('2026-27',a.loc['2026':'2027']),('2027-28',a.loc['2027':'2028'])]:
  if len(x): print(label,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(ddof=1),6))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280128_risk_adjusted_trend_signal.csv',index=False)
