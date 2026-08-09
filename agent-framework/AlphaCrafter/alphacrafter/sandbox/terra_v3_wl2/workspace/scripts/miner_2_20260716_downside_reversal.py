import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); r=p.pct_change()
# Downside-asymmetric reversal: recent 5d loss scaled by trailing downside deviation.
down=r.where(r<0,0).rolling(40,min_periods=20).std()
f=-r.rolling(5,min_periods=5).sum()/down.replace(0,np.nan)
ics=[]; ns=[]; cov=[]
for d in f.index:
 z=pd.concat([f.loc[d],r.shift(-1).loc[d]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); cov.append(len(z)/15)
ic=np.array(ics)
print('dates',len(ic),'avg_n',np.mean(ns),'meanIC',ic.mean(),'absIC',abs(ic.mean()),'std',ic.std(ddof=1),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'coverage',np.mean(cov))
for h in [5,10]:
 ii=[]
 for d in f.index:
  z=pd.concat([f.loc[d],r.shift(-h).loc[d]],axis=1).dropna()
  if len(z)>=8: ii.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(ii), 'dates',len(ii))
rk=f.rank(axis=1,pct=True); print('turnover',rk.diff().abs().mean(axis=1).dropna().mean())
print('recent250',ic[-250:].mean(),'recent_icir',ic[-250:].mean()/ic[-250:].std(ddof=1))
