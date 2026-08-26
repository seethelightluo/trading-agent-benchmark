import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-12-25')
px={}
for f in glob.glob('../persistent/stock_data/*.csv'):
    s=f.rsplit('/',1)[-1][:-4]; d=pd.read_csv(f); d.date=pd.to_datetime(d.date)
    px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Cross-asset relative short reversal: reverse each asset's 5d return
# relative to the same-day cross-sectional median, scaled by trailing volatility.
ret5=p.pct_change(5); csmed=ret5.median(axis=1)
vol20=r.rolling(20).std()*np.sqrt(252)
f=(-(ret5.sub(csmed,axis=0))).div(vol20).shift(1)
print('assets',len(p.columns),'dates',len(p),'cut',p.index.max().date())
for h in [1,5,10,20]:
    rr=p.pct_change(h).shift(-h); a=[]; ns=[]
    for dt in p.index:
        z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
    a=np.asarray(a)
    print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
    if h==10: print('recent500',a[-500:].mean(),a[-500:].mean()/a[-500:].std(ddof=1))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20341225_relative_reversal_signal.csv')
