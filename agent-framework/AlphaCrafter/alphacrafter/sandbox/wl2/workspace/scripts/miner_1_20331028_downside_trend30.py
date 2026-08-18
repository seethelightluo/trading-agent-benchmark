import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 P[s]=d['close'].replace(0,np.nan)
px=pd.DataFrame(P)
ret=np.log(px).diff()
# Candidate: medium trend divided by downside volatility, lagged
mom=np.log(px/px.shift(30))
down=ret.where(ret<0).rolling(30,min_periods=15).std()
sig=(mom/down).shift(1)
fwd=np.log(px.shift(-10)/px)
ics=[]; dates=[]; nobs=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); nobs.append(ok.sum())
ics=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
print('candidate downside_adjusted_trend_30d_h10')
print('dates',len(ics),'avg_n',np.mean(nobs),'assets',len(U))
print('IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1), (ics>0).mean()))
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-12-31')]:
 z=ics.loc[a:b]; print(a,b,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
# coverage and turnover proxy rank changes daily among valid
print('coverage',sig.notna().sum(axis=1).mean()/len(U),'signal coverage',sig.notna().stack().mean())
r=sig.rank(axis=1,pct=True); turn=(r-r.shift(1)).abs().mean(axis=1).dropna(); print('turnover_proxy',turn.mean())
for h in [5,10,20]:
 yy=np.log(px.shift(-h)/px); z=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[dt][ok],yy.loc[dt][ok]).statistic)
 print('h',h,'ic',np.nanmean(z),'n',len(z))
# save artifact
out=pd.DataFrame({'date':np.repeat(sig.index,len(U)),'symbol':U*len(sig.index),'signal':sig.to_numpy().ravel()})
out.to_csv('scripts/miner_1_20331028_downside_trend30_signal.csv',index=False)
