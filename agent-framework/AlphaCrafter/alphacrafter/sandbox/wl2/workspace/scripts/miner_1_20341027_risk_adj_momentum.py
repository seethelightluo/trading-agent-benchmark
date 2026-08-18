import pandas as pd, numpy as np, os
from scipy.stats import pearsonr

SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2034-10-26')
D={}
for s in SYMS:
    x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
    x=x.loc[x.index<=END]
    # daily decimal returns; use close, lag signal by one completed day
    r=x.close.pct_change()
    mom=x.close.pct_change(20)
    vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
    D[s]=pd.DataFrame({'f':(mom/vol).shift(1),'close':x.close})
idx=sorted(set().union(*[set(v.index) for v in D.values()]))
F=pd.DataFrame({s:D[s].f for s in SYMS}).reindex(idx)
C=pd.DataFrame({s:D[s].close for s in SYMS}).reindex(idx)
# each asset forward return from t close to t+h close; signal is already lagged
out=[]
for h in [1,5,10,20,40]:
    fr=C.shift(-h)/C-1
    vals=[]; ns=[]; dates=[]
    for dt in F.index:
        z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
            ns.append(len(z)); dates.append(dt)
    a=np.array(vals); mean=np.nanmean(a); sd=np.nanstd(a,ddof=1)
    print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(mean,6),'ICIR',round(mean/sd,6),'hit',round(np.mean(a>0),4))
    for lo,hi in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2032-12-31'),('2033','2034-10-26')]:
        m=(pd.Series(dates)>=pd.Timestamp(lo))&(pd.Series(dates)<=pd.Timestamp(hi)); aa=a[m.values]
        print(' ',lo, 'n',len(aa),'icir',round(np.nanmean(aa)/np.nanstd(aa,ddof=1),5) if len(aa)>1 else None)
    out.append((h,a,dates))
# coverage and rank turnover
valid=F.notna().sum(axis=1)>=8
coverage=F.notna().sum().sum()/(len(F)*len(SYMS))
ranks=F.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)).median()
print('coverage',round(coverage,5),'dates',len(F),'turnover_proxy',round(turn,5),'assets',len(SYMS))
# signal artifact for admission provenance
sig=F.loc[valid].copy(); sig.index.name='date'; sig.reset_index().to_csv('../persistent/miner_1_20341027_risk_adj_momentum_signal.csv',index=False)
