import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
B='../persistent/stock_data'; P={}
for s in U:
    f=os.path.join(B,s+'.csv')
    if os.path.exists(f):
        d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
        P[s]=d['close'].astype(float)
pdpx=pd.DataFrame(P).sort_index().ffill(); cutoff=pd.Timestamp('2031-01-22'); px=pdpx.loc[:cutoff]
r=px.pct_change(); vol=r.rolling(20,min_periods=15).std(); mom10=px.pct_change(10); mom40=px.pct_change(40)
# Trend-confirmed, volatility-scaled short momentum. A positive 40d market breadth
# activates continuation; a weak breadth regime activates a bounded 10d reversal.
breadth=(mom40>0).mean(axis=1)
market=(px.pct_change(20).mean(axis=1)>0)
f_cont=(mom10/(vol+1e-8)).clip(-8,8)
f_rev=(-mom10/(vol+1e-8)).clip(-8,8)
f=f_cont.where((breadth>=0.5)&market, f_rev)
print('sample',px.index.min().date(),px.index.max().date(),'assets',px.shape[1],'cutoff',cutoff.date())
for h in [5,10,20]:
    y=px.shift(-h).div(px)-1; ic=[]; ns=[]; dates=[]; turns=[]
    prev=None
    for dt in f.index:
        z=pd.concat([f.loc[dt].rename('x'),y.loc[dt].rename('y')],axis=1).dropna()
        if len(z)>=8:
            ic.append(spearmanr(z.x,z.y).statistic); ns.append(len(z)); dates.append(dt)
            if prev is not None:
                q=pd.concat([f.loc[prev],f.loc[dt]],axis=1).dropna()
                turns.append((q.iloc[:,1].rank(pct=True)-q.iloc[:,0].rank(pct=True)).abs().mean())
        prev=dt
    a=np.asarray(ic); sd=a.std(ddof=1)
    print('horizon',h,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),8),'ICIR_ann',round(a.mean()/sd*np.sqrt(252),5),'hit',round(np.mean(a>0),5),'turnover',round(np.mean(turns),6))
    if h==10:
        for yr in sorted(set(d.year for d in dates)):
            q=a[[d.year==yr for d in dates]]
            if len(q): print(' regime',yr,'n',len(q),'IC',round(q.mean(),6))
