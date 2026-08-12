import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root=Path('../persistent/stock_data')
def load(s): return pd.read_csv(root/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'].loc[:'2026-07-15']
p=pd.concat([load(s).rename(s) for s in syms],axis=1,sort=True)
r=p.pct_change(fill_method=None); m=r.mean(axis=1,skipna=True)
betas={}
for s in syms:
 x=pd.concat([r[s],m],axis=1).dropna(); x.columns=['x','m']
 cov=x['x'].rolling(60,min_periods=45).cov(x['m']); var=x['m'].rolling(60,min_periods=45).var()
 betas[s]=(cov/var).reindex(p.index)
beta=pd.DataFrame(betas)
# 20d total return less rolling-beta times 20d benchmark total return
mr=m.rolling(20,min_periods=20).sum()
asset20=p.pct_change(20,fill_method=None)
fac=asset20-beta.mul(mr,axis=0)
fwd={h:p.pct_change(h,fill_method=None).shift(-h) for h in [1,5,10]}
def ev(y,idx=None):
 q=[]; ns=[]
 for dt in (fac.index if idx is None else idx):
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=np.asarray(q)
 return len(q),round(float(np.mean(ns)),2),round(float(np.mean(ns)/15),4),round(float(q.mean()),5),round(float(q.mean()/q.std(ddof=1)),5),round(float(np.mean(q>0)),4)
print('valid_beta',int(beta.notna().sum().sum()),'valid_factor',int(fac.notna().sum().sum()))
for h in [1,5,10]: print(str(h)+'d',ev(fwd[h]))
print('turn',round(float(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),5),'corr20',round(float(fac.stack().corr(asset20.stack())),4),'corrrev',round(float(fac.stack().corr((-p.pct_change(5,fill_method=None)).stack())),4))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]: print(a+'-'+b,ev(fwd[1],fac.loc[a:b].index))
