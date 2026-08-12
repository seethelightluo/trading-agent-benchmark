import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# Recovery from intermediate drawdown: distance above 60-session low, lagged one day.
C=pd.DataFrame({s:D[s].close for s in U}).sort_index(); F=(C/C.rolling(60,min_periods=40).min()-1).shift(1)
Y={h: C.shift(-h).div(C)-1 for h in [1,5,10]}
outs={}
for h in Y:
 q=[];ns=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y[h].loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.array(q);outs[h]=q
 print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for yr in range(2020,2027):
 x=[]
 for dt in F.loc[str(yr)].index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y[1].loc[dt]}).dropna()
  if len(z)>=8:x.append(spearmanr(z.f,z.y).statistic)
 print('regime',yr,'dates',len(x),'IC',round(np.mean(x),6) if x else None,'ICIR',round(np.mean(x)/np.std(x,ddof=1),4) if len(x)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for k in [252,504,756]:
 x=outs[1][-k:];print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'dates',len(x))
print('corr_existing_mom',round(pd.concat([F.stack(),C.pct_change().rolling(20).sum().stack()],axis=1).dropna().iloc[:,0].rank().corr(pd.concat([F.stack(),C.pct_change().rolling(20).sum().stack()],axis=1).dropna().iloc[:,1].rank()),4))
