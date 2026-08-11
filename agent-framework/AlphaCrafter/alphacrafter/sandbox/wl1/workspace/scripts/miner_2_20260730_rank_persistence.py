import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
def load(s):
    return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut].close
p=pd.concat({s:load(s) for s in U},axis=1).sort_index()
ret20=p.pct_change(20); ret60=p.pct_change(60)
f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i in range(60,len(p)):
    a=ret20.iloc[i]; b=ret60.iloc[i]
    f.iloc[i]=((a.rank(pct=True)+b.rank(pct=True))/2*np.where(np.sign(a)==np.sign(b),1.0,0.5))
for hor in [1,5,10,20]:
    ics=[]; ns=[]; dates=[]; turns=[]; prev=None
    for i in range(60,len(p)-hor):
        q=pd.concat([f.iloc[i],(p.iloc[i+hor]/p.iloc[i]-1).rename('y')],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,0].nunique()>1:
            ics.append(spearmanr(q.iloc[:,0],q.y).statistic); ns.append(len(q)); dates.append(p.index[i])
            ranks=q.iloc[:,0].rank(pct=True)
            if prev is not None: turns.append(np.mean(np.abs(ranks-prev.reindex(ranks.index))))
            prev=ranks
    x=np.array(ics)
    print('horizon',hor,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'turnover',round(np.mean(turns),6))
    if hor==10:
        yy=np.array([d.year for d in dates]); print('regime',{int(y):round(x[yy==y].mean(),6) for y in sorted(set(yy))})
print('assets',len(U),'range',p.index[0].date(),p.index[-1].date())
