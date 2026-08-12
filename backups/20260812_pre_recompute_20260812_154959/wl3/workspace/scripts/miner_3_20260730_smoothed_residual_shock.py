import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
R=pd.concat({s:D[s].close.pct_change() for s in U},axis=1).sort_index(); X=R.sub(R.median(axis=1),axis=0)
# Novel lower-turnover residual shock: 5-day accumulated relative move, normalized by 60d residual vol, then 3d EMA; contrarian
raw=-X.rolling(5,min_periods=5).sum()/(X.rolling(60,min_periods=30).std()*np.sqrt(5)+1e-12)
F=raw.ewm(span=3,min_periods=3,adjust=False).mean()
y=R.shift(-1); obs=[]; alln=0
for d in F.index:
 a=pd.DataFrame({'f':F.loc[d],'y':y.loc[d]}).dropna(); alln+=len(a)
 if len(a)>=8 and a.f.nunique()>1: obs.append((d,spearmanr(a.f,a.y).statistic,len(a)))
O=pd.DataFrame(obs,columns=['date','ic','n']); q=O.ic
turn=F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2
print('dates',len(q),'avgN',O.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',alln/(len(F)*15),'turn',turn)
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=O[(O.date.dt.year>=lo)&(O.date.dt.year<=hi)].ic;print('regime',lo,hi,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'N',len(z))
for h in [5,10]:
 yy=pd.concat({s:D[s].close.pct_change(h).shift(-h) for s in U},axis=1); zz=[]
 for d in F.index:
  a=pd.DataFrame({'f':F.loc[d],'y':yy.loc[d]}).dropna()
  if len(a)>=8:zz.append(spearmanr(a.f,a.y).statistic)
 zz=np.array(zz);print('decay',h,'IC',zz.mean(),'ICIR',zz.mean()/zz.std(ddof=1),'N',len(zz))
