import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr

symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in symbols:
    f=f'{base}/{s}.csv'
    if os.path.exists(f):
        d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
        px[s]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# equal-weight market return, and VIX macro signal (observation only)
market=r.mean(axis=1,skipna=True)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
vix=vix.reindex(p.index).ffill()
# residual 20d return: trailing 60d beta to market, all inputs through t, signal at t then forward t+1
betas=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
for s in p:
    cov=r[s].rolling(60,min_periods=40).cov(market)
    var=market.rolling(60,min_periods=40).var()
    betas[s]=cov/var.replace(0,np.nan)
resid=r.sub(betas.mul(market,axis=0),axis=0)
res20=np.expm1(np.log1p(resid.clip(-0.99)).rolling(20,min_periods=20).sum())
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
raw=(res20/vol.replace(0,np.nan)).clip(-5,5)
# macro conditioning: retain residual momentum in calm/up-VIX-normalizing regimes; damp in stress
vmed=vix.rolling(60,min_periods=40).median()
# continuous stress multiplier, using only lagged completed data; not a directional lookahead
stress=(vix/vmed).clip(0.5,2.0)
factor=raw.div(stress**0.5,axis=0)
factor=factor.shift(1)
fwd=r.shift(-1)
rows=[]
for dt in factor.index:
    x=factor.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        rows.append((dt,ic,len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(out),'avgN',out.n.mean(),'coverage',out.n.sum()/(len(out)*15))
print('IC %.8f ICIR %.8f hit %.4f'%(out.ic.mean(),out.ic.mean()/out.ic.std(ddof=1), (out.ic>0).mean()))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020-01-01','2021-12-31'),('2022-01-01','2023-12-31'),('2024-01-01','2025-12-31'),('2026-01-01','2027-12-31')]:
 q=out.loc[a:b].ic; print(a,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 yy=(1+r).rolling(h).apply(np.prod,raw=True).shift(-h)
 rr=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 rr=pd.Series(rr).dropna(); print('h',h,'IC',rr.mean(),'ICIR',rr.mean()/rr.std(ddof=1),'n',len(rr))
print('max library corr unavailable: candidate is residual momentum, compare manually')
