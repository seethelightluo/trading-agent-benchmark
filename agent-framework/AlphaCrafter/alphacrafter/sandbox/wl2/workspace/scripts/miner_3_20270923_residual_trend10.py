import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={}
for s in S:
 f=f'{base}/{s}.csv'
 if os.path.exists(f): px[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); m=r.mean(axis=1,skipna=True)
# 10d market-residual trend, scaled by idiosyncratic volatility; all inputs lagged one day
v=m.rolling(40,min_periods=25).var()
b=pd.DataFrame({s:r[s].rolling(40,min_periods=25).cov(m)/v.replace(0,np.nan) for s in p})
e=r.sub(b.mul(m,axis=0),axis=0)
trend=np.log1p(e.clip(-.99)).rolling(10,min_periods=10).sum()
idvol=e.rolling(20,min_periods=15).std()
f=(trend/idvol.replace(0,np.nan)).clip(-8,8).shift(1)
# reward persistent, not just one-day residuals
f=f.ewm(span=3,min_periods=3,adjust=False).mean()
fwd=r.shift(-1); rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(out),'avgN',out.n.mean(),'coverage',out.n.sum()/(len(out)*15))
print('IC %.8f ICIR %.8f hit %.4f'%(out.ic.mean(),out.ic.mean()/out.ic.std(ddof=1),(out.ic>0).mean()))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,bx in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 q=out.loc[a:bx].ic; print('regime',a,bx,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 yy=(1+r).rolling(h).apply(np.prod,raw=True).shift(-h); rr=[]
 for d in f.index:
  z=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 rr=pd.Series(rr); print('h',h,'IC',rr.mean(),'ICIR',rr.mean()/rr.std(ddof=1),'dates',len(rr))
print('max_abs_library_correlation unavailable; residualized construction targets low overlap')
