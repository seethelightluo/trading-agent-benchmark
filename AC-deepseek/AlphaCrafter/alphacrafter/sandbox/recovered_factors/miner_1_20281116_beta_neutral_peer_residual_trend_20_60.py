# miner_1, one candidate: beta-neutral peer-relative residual trend, price-only
# At each t, score is the preceding 20d asset return less its lagged 60d beta times
# the median cross-asset return, divided by lagged residual-return volatility.
import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2028-11-15')
close={}
for a in assets:
    d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
    close[a]=d.loc[:cutoff,'close']
C=pd.DataFrame(close).sort_index(); R=C.pct_change(); peer=R.median(axis=1)
# beta(t) is based on observations ending t-1
beta=R.rolling(60,min_periods=45).cov(peer).div(peer.rolling(60,min_periods=45).var(),axis=0).shift(1)
resid=R.sub(beta.mul(peer,axis=0))
# both trend endpoints are t-1; scale has only data through t-1
excess20=(1+R).rolling(20,min_periods=20).apply(np.prod,raw=True).shift(1)-1
peer20=(1+peer).rolling(20,min_periods=20).apply(np.prod,raw=True).shift(1)-1
rvol=resid.rolling(60,min_periods=45).std().shift(1)
F=excess20.sub(beta.mul(peer20,axis=0)).div(rvol).replace([np.inf,-np.inf],np.nan)
print('candidate: lagged 20d beta-neutral cross-asset residual trend / lagged 60d residual volatility')
print('visible cutoff',cutoff.date(),'panel rows',len(F),'valid cells',F.notna().sum().sum(),'/',F.size,'coverage %.4f'%(F.notna().sum().sum()/F.size),'mean names %.2f'%F.notna().sum(axis=1).mean())
ranks=F.rank(axis=1,pct=True);print('mean daily rank-change turnover %.6f'%ranks.diff().abs().stack().mean(),'mean cs std %.6f'%F.std(axis=1).mean())
metrics={}
for h in [1,5,10,20]:
    fw=(1+R).rolling(h,min_periods=h).apply(np.prod,raw=True).shift(-h)
    vals=[]; breadth=[]
    for d in F.index:
        x=pd.concat([F.loc[d],fw.loc[d]],axis=1).dropna()
        if len(x)>=8:
            ic=spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic
            if np.isfinite(ic): vals.append((d,ic));breadth.append(len(x))
    z=np.array([q for _,q in vals]); metrics[h]=vals
    print('H%d dates=%d IC=%.6f ICIR=%.6f hit=%.4f breadth=%.2f'%(h,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),np.mean(breadth)))
for name,start,end in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027-now','2027-01-01','2028-11-15'),('recent180','2028-05-19','2028-11-15')]:
    z=np.array([q for d,q in metrics[10] if pd.Timestamp(start)<=d<=pd.Timestamp(end)])
    print('%s H10 dates=%d IC=%s ICIR=%s hit=%s'%(name,len(z),'%.6f'%z.mean() if len(z) else 'NA','%.6f'%(z.mean()/z.std(ddof=1)) if len(z)>1 else 'NA','%.4f'%(z>0).mean() if len(z) else 'NA'))
