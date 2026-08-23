import pandas as pd, numpy as np, os
from scipy.stats import spearmanr

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
def load(sym):
    x=pd.read_csv(os.path.join(base,sym+'.csv'))
    x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
    return x['close'].astype(float)
px=pd.concat({a:load(a) for a in assets},axis=1)
dxy=pd.read_csv('../persistent/index_data/DXY.csv'); dxy['date']=pd.to_datetime(dxy['date']); dxy=dxy.set_index('date').sort_index()['close'].astype(float)
allx=pd.concat([px,dxy.rename('DXY')],axis=1).sort_index().ffill()
r=allx.pct_change(); dr=r['DXY']
# residual trend: trailing 20d asset return minus rolling 60d beta to DXY times DXY 20d return
beta=r[assets].rolling(60,min_periods=45).cov(dr).div(dr.rolling(60,min_periods=45).var(),axis=0)
trend=px[assets].pct_change(20)-beta.mul(dxy.pct_change(20),axis=0)
# only dates for which forward return is observable, data ends 2026-07-15
out=[]
for h in [1,5,10]:
    fwd=px[assets].shift(-h)/px[assets]-1
    ics=[]; turns=[]; nms=[]
    for dt in trend.index:
        v=trend.loc[dt]; y=fwd.loc[dt]; ok=v.notna()&y.notna()
        if ok.sum()>=8:
            ics.append(spearmanr(v[ok],y[ok]).statistic); nms.append(ok.sum())
            if len(ics)>1:
                prev=trend.loc[trend.index[trend.index.get_loc(dt)-1]]
                p=prev.notna()&ok
                turns.append((v[p].rank(pct=True)-prev[p].rank(pct=True)).abs().mean())
    z=np.array(ics); out.append((h,len(z),np.mean(nms),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0),np.mean(turns) if turns else np.nan))
print('factor=dxy_residual_momentum_20d universe=15')
for x in out: print('h dates avgN IC ICIR hit turnover',x)
print('coverage',trend.notna().sum(axis=1).mean()/15)
# correlations against known simple signal artifacts, pooled date-name ranks
known={'mom':px[assets].pct_change(20),'rev':-px[assets].pct_change(5),'peer':r[assets].sub(r[assets].median(axis=1),axis=0)}
for k,v in known.items():
    z=pd.concat([trend.stack().rename('a'),v.stack().rename('b')],axis=1).dropna(); print('corr',k,z.corr(method='spearman').iloc[0,1])
print('period',trend.index.min().date(),trend.index.max().date())
