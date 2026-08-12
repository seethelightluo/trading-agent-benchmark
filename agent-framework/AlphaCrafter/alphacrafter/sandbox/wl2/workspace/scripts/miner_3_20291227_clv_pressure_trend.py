import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in ASSETS:
    x=get_stock_daily_data(s,days=2600)
    if x is None: continue
    x=x.sort_values('date').drop_duplicates('date').set_index('date')
    D[s]=x
# Candidate: 5-day close-location pressure, demeaned and volatility normalized;
# reversal of weak closes is expected to mean revert cross-sectionally.
rows=[]
for s,x in D.items():
    c=x.close.astype(float); h=x.high.astype(float); l=x.low.astype(float)
    r=c.pct_change()
    clv=((2*c-h-l)/(h-l).replace(0,np.nan)).rolling(5).mean()
    for i in range(65,len(x)-10):
        dt=x.index[i]
        # only completed data at i; forward returns i+1 onward
        vol=r.iloc[i-19:i+1].std()
        if not np.isfinite(vol) or vol<1e-5: continue
        trend=c.iloc[i]/c.iloc[i-20]-1
        pressure=clv.iloc[i]
        # weak close pressure, but favor established positive 20d trend (interpretable conditional)
        f=(-pressure + 0.35*trend/(vol+0.01))/(vol+0.01)
        fwd1=c.iloc[i+1]/c.iloc[i]-1
        fwd5=c.iloc[i+5]/c.iloc[i]-1
        rows.append((dt,s,f,fwd1,fwd5))
z=pd.DataFrame(rows,columns=['date','s','f','r1','r5'])
ics=[]; ic5=[]; cov=[]; turnovers=[]
for dt,g in z.groupby('date'):
    if len(g)>=8:
        a=g.f.corr(g.r1,method='spearman'); b=g.f.corr(g.r5,method='spearman')
        if np.isfinite(a): ics.append(a); ic5.append(b)
        cov.append(len(g)/15)
# turnover rank changes sequentially
good=z[z.date.isin([d for d,g in z.groupby('date') if len(g)>=8])]
for a,b in zip(sorted(good.date.unique())[:-1],sorted(good.date.unique())[1:]):
    x=good[good.date==a].set_index('s').f; y=good[good.date==b].set_index('s').f
    q=pd.concat([x,y],axis=1).dropna()
    if len(q)>=8: turnovers.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
for name,v in [('daily',ics),('5d',ic5)]:
    v=np.asarray(v,float); print(name,'IC %.5f ICIR %.5f hit %.3f n_dates %d'%(np.nanmean(v),np.nanmean(v)/np.nanstd(v,ddof=1),np.mean(v>0),len(v)))
print('dates',len(ics),'avg instruments',np.mean([len(g) for _,g in z.groupby('date') if len(g)>=8]),'coverage',np.mean(cov),'turnover_proxy',1-np.nanmean(turnovers))
print('recent',pd.DataFrame({'date':sorted(good.date.unique())[-60:]}).tail(1).date.iloc[0])
