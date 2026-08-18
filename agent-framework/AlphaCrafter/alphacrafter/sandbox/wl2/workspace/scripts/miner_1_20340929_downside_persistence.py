import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in watch:
 f='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
 px[s]=d['close'].replace(0,np.nan)
P=pd.DataFrame(px).loc[:'2034-09-28']
r=P.pct_change()
# Downside-adjusted intermediate momentum: recent 20d return divided by downside deviation,
# with a trend persistence gate based on sign agreement of 5d and 20d returns.
down=r.where(r<0,0).rolling(20,min_periods=15).std()
ret20=P/P.shift(20)-1
ret5=P/P.shift(5)-1
agree=((ret5>0)==(ret20>0)).astype(float)
f=(ret20/(down*np.sqrt(252)+1e-8))*agree
# lag: signal at t uses through t-1
f=f.shift(1)
rows=[]
for h in [1,3,5,10,20]:
 fr=P.shift(-h)/P-1
 ics=[]; dates=[]; nobs=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); nobs.append(len(z))
 ic=pd.Series(ics,index=dates)
 print(h,'dates',len(ic),'avg_n',round(np.mean(nobs),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 for lo,hi in [('2020','2025'),('2026','2028'),('2029','2031'),('2032','2034')]:
  q=ic.loc[lo:hi]
  print(' regime',lo,hi,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
# coverage and turnover rank
valid=f.notna().sum(axis=1)/15
rank=f.rank(axis=1,pct=True)
turn=rank.diff().abs().mean(axis=1).mean()
print('coverage',round(valid.mean(),4),'turnover_proxy',round(turn,6),'period',P.index.min(),P.index.max())
