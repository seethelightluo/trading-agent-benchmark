import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
    try: d=get_index_daily_data(s,days=4100)
    except Exception: d=get_stock_daily_data(s,days=4100)
    if d is not None and len(d):
        d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').close.astype(float).sort_index()
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
# Residual medium-term leadership: 40d return stripped of rolling beta to equal-weight market,
# scaled by idiosyncratic volatility. Lag before forward-return tests.
win=60
cov=r.rolling(win).cov(m); var=m.rolling(win).var(); beta=cov.div(var.replace(0,np.nan),axis=0)
res=r.sub(beta.mul(m,axis=0),axis=0)
resvol=res.rolling(40).std()*np.sqrt(252)
raw=(p/p.shift(40)-1) - beta*(p.div(p.shift(40))-1).sub(m.rolling(40).sum(),axis=0)
# equivalent residual return approximation, normalized for risk
sig=raw.div(resvol.replace(0,np.nan)).shift(1)
rank=sig.rank(axis=1,pct=True)
fw={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
for h,f in fw.items():
    vals=[]; dates=[]; ns=[]
    for dt in rank.index:
        z=pd.concat([rank.loc[dt],f.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
    x=pd.Series(vals,index=dates).dropna()
    print('%dd dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
    if h in (1,10):
        for yr,g in x.groupby(x.index.year): print(' year',yr,'IC=%.6f n=%d'%(g.mean(),len(g)))
print('dates',len(p),'assets',p.shape[1],'avg_names',p.notna().sum(axis=1).mean(),'coverage',sig.notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).dropna().mean())
rank.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20310206_residual_leadership_signal.csv',index=False)
