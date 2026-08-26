import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2033-09-04']; r=px.pct_change()
vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
sig=((r.rolling(20,min_periods=15).sum()-r.rolling(60,min_periods=45).sum())/(vol+1e-12)).shift(1)
sig.to_csv('scripts/miner_1_20330905_volscaled_acceleration_signal.csv')
print('assets',len(A),'dates',len(px),'cutoff',px.index[-1].date(),'coverage',round(sig.notna().mean().mean(),4))
for h in [1,5,10,20]:
    f=px.shift(-h)/px-1; vals=[]; ns=[]
    for dt in px.index:
        q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
        if len(q)>=8:
            z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
            if np.isfinite(z): vals.append(z); ns.append(len(q))
    a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4),'thirds',[round(float(x.mean()),6) for x in np.array_split(a,3)])
for n in [180,500,750]:
    vals=[]; f=px.shift(-10)/px-1
    for dt in sig.index[-n:]:
        q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
        if len(q)>=8:
            z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
            if np.isfinite(z): vals.append(z)
    a=np.array(vals); print('recent',n,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
ranks=sig.rank(axis=1,pct=True); print('turnover',round(ranks.diff().abs().stack().mean(),6),'avgN',round(sig.notna().sum(axis=1).mean(),2))
