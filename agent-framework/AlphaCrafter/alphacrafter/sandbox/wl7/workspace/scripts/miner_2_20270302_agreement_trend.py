import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2027-03-02')
D={}
for s in U:
    p=Path('../persistent/stock_data')/(s+'.csv')
    x=pd.read_csv(p); x.date=pd.to_datetime(x.date).dt.normalize()
    D[s]=x.drop_duplicates('date').set_index('date').sort_index().close.astype(float).loc[:CUT]
P=pd.concat(D,axis=1).sort_index(); R=P.pct_change()
# Multi-horizon agreement: medium trend, only rewarded when 5/20/60d directions agree;
# scale by 20d realized volatility. The one-session shift enforces observability.
rows=[]
for s in U:
    p=P[s]; r=R[s]
    r5=p.pct_change(5); r20=p.pct_change(20); r60=p.pct_change(60)
    agree=(np.sign(r5)+np.sign(r20)+np.sign(r60))/3.0
    f=(agree*r20/(r.rolling(20,min_periods=15).std()+1e-12)).shift(1)
    for h in [1,5,10,20]:
        fr=p.shift(-h).div(p).sub(1)
        rows.append(pd.DataFrame({'date':P.index,'asset':s,'f':f.values,'fr':fr.values,'h':h}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(g):
    z=[]; ns=[]
    for _,x in g.groupby('date'):
        if len(x)>=8 and x.f.nunique()>1 and x.fr.nunique()>1:
            z.append(x.f.corr(x.fr,method='spearman')); ns.append(len(x))
    z=pd.Series(z)
    return {'dates':len(z),'avg_n':round(float(np.mean(ns)),2),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),6),'hit':round(float((z>0).mean()),4)}
print('cutoff',CUT.date(),'assets',len(D),'calendar_dates',P.index.nunique(),'rows',len(q),'coverage',round(q[q.h==1].shape[0]/(P.index.nunique()*len(U)),4))
for h in [1,5,10,20]: print('horizon',h,stats(q[q.h==h]))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.h==1)&(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q[q.h==1].pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True)
print('turnover',round(float(r.diff().abs().mean().mean()),6))
q[q.h==1].to_csv('scripts/miner_2_20270302_agreement_trend_signal.csv',index=False)
