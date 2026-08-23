import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict().get('watch_list',[])
if not U: U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    d=get_stock_daily_data(s,days=4000)
    if d is not None and len(d):
        d=d.copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
        D[s]=d
# Candidate: 3d opening-gap reversal scaled by trailing 20d close-return volatility, lagged at date t
# At date t use data through t; forward return is t+1 close/close - 1 (signal is formed after t close).
rows=[]
for s,d in D.items():
    gap=d.open/d.close.shift(1)-1
    vol=d.close.pct_change().rolling(20,min_periods=15).std()
    sig=-(gap.rolling(3,min_periods=3).mean()/vol)
    fr=d.close.shift(-1)/d.close-1
    z=pd.DataFrame({'date':d.index,'sig':sig,'fr':fr,'s':s}).dropna()
    rows.append(z)
x=pd.concat(rows,ignore_index=True)
ics=[]; counts=[]
for dt,g in x.groupby('date'):
    if len(g)>=8 and g.sig.nunique()>1 and g.fr.nunique()>1:
        ics.append(g.sig.corr(g.fr,method='spearman')); counts.append(len(g))
ics=np.array(ics); print('dates',len(ics),'avg_n',np.mean(counts),'coverage',len(x)/sum(len(d) for d in D.values()))
print('IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0))
# horizons using forward close returns h sessions per asset
for h in [5,10]:
    rr=[]
    for s,d in D.items():
        gap=d.open/d.close.shift(1)-1; vol=d.close.pct_change().rolling(20,min_periods=15).std()
        sig=-(gap.rolling(3,min_periods=3).mean()/vol)
        fr=d.close.shift(-h)/d.close-1
        rr.append(pd.DataFrame({'date':d.index,'sig':sig,'fr':fr}).dropna().assign(s=s))
    q=pd.concat(rr).reset_index(drop=True)
    z=[]
    for dt,g in q.groupby('date'):
        if len(g)>=8: z.append(g.sig.corr(g.fr,method='spearman'))
    z=np.array(z); print(h,'dates',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1))
# save signal artifact
x.to_csv('scripts/miner_3_20261022_volscaled_gap_signal.csv',index=False)
