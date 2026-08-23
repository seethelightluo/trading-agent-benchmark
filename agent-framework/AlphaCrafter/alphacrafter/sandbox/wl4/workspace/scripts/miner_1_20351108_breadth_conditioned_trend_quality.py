import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); mom=P/P.shift(20)-1; vol=R.rolling(20).std()*np.sqrt(252)
base=mom/vol.replace(0,np.nan)
valid=R.notna().rolling(20).sum().sum(axis=1).replace(0,np.nan)
breadth=(R.rolling(20).sum()>0).sum(axis=1)/valid
F=base.multiply(1+0.65*(breadth-0.5)*2,axis=0).shift(1)
rows=[]
for h in [5,10,20]:
 fwd=P.shift(-h)/P-1
 for dt in F.index:
  z=pd.concat([F.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
D=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('dates',P.index.min(),P.index.max(),'assets',P.shape[1])
for h in [5,10,20]:
 q=D[D.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1); print('h',h,'obs',len(q),'avg_n',q.n.mean(),'IC',m,'ICIR',m/sd*np.sqrt(252),'hit',(q.ic>0).mean(),'coverage',q.n.mean()/15)
 for n in [120,260,520,780]:
  z=q.tail(n); print(' recent',n,'ICIR',z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252),'IC',z.ic.mean())
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
