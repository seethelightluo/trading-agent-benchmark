import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-03-06'); base='../persistent/stock_data'; frames={}
for s in U:
    d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date)
    d=d[d.date<=cut].set_index('date')
    frames[s]=d[['open','high','low','close','volume']].astype(float)
# Volume-weighted directional pressure: close location value times abnormal volume,
# aggregated over 10 sessions and lagged one completed session.
clv={}; vs={}
for s,d in frames.items():
    rng=(d.high-d.low).replace(0,np.nan)
    clv[s]=((2*d.close-d.high-d.low)/rng).clip(-1,1)
    med=d.volume.rolling(20,min_periods=10).median()
    vs[s]=np.log1p(d.volume/(med+1e-12)).clip(-3,3)
pressure=pd.DataFrame({s:clv[s]*vs[s] for s in U}).rolling(10,min_periods=8).sum().shift(1)
P=pd.DataFrame({s:frames[s].close for s in U}).sort_index(); print('rows',len(P),'range',P.index.min().date(),P.index.max().date(),'cut',cut.date())
for h in [5,10,20]:
    f=P.shift(-h)/P-1; vals=[]; ns=[]
    for dt in P.index:
        z=pd.concat([pressure.loc[dt],f.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
            if np.isfinite(q): vals.append(q); ns.append(len(z))
    a=np.asarray(vals); ic=a.mean(); icir=ic/(a.std(ddof=1)+1e-12)*np.sqrt(len(a))
    print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'minN',min(ns),'IC',round(ic,6),'absIC',round(abs(ic),6),'ICIR',round(icir,6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4))
    if len(a)>=250:
        q=a[-250:]; print('recent250',round(q.mean(),6),'recentICIR',round(q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(250),6))
r=pressure.rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean(axis=1).dropna().mean(),6),'panel_valid',round(pressure.notna().sum().sum()/pressure.size,4))
