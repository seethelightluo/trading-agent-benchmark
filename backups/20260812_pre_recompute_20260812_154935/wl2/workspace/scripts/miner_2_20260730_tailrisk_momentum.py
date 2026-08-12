import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
C=pd.DataFrame({s:D[s].close for s in U}).sort_index(); R=C.pct_change()
# Tail-risk-adjusted momentum: trailing 20-session return divided by downside deviation,
# with the complete signal lagged one session to prevent look-ahead.
down=R.where(R<0,0).rolling(20,min_periods=15).std()
F=(R.rolling(20,min_periods=15).sum()/down.replace(0,np.nan)).shift(1)
Y=C.shift(-1).div(C)-1
q=[]; ns=[]; dates=[]
for dt in F.index:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:
  q.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); dates.append(dt)
q=np.asarray(q)
print('horizon 1 dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for yr in range(2020,2027):
 x=q[[d.year==yr for d in dates]]
 if len(x)>1: print('regime',yr,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [5,10]:
 Yh=C.shift(-h).div(C)-1; zq=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Yh.loc[dt]}).dropna()
  if len(z)>=8:zq.append(spearmanr(z.f,z.y).statistic)
 zq=np.asarray(zq);print('horizon',h,'dates',len(zq),'IC',round(zq.mean(),6),'ICIR',round(zq.mean()/zq.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
# library overlap proxies
for n,x in {'mom20':R.rolling(20).sum(),'rev5':-R.rolling(5).sum(),'vol20':-R.rolling(20).std()}.items():
 z=pd.concat([F.stack(),x.stack()],axis=1).dropna(); print('corr',n,round(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),4))
