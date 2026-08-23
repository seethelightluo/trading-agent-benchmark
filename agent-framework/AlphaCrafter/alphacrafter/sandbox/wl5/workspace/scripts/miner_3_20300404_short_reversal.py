import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d)>100:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
        frames[s]=d['close'].astype(float)
px=pd.DataFrame(frames).sort_index().ffill()
# factor: inverse short-term return, scaled by trailing volatility, with cross-sectional demeaning
ret5=px.pct_change(5); vol20=px.pct_change().rolling(20).std()*np.sqrt(252)
f=(-ret5/vol20).replace([np.inf,-np.inf],np.nan)
# rank-style cross-sectional demean preserves interpretable signal
f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for h in [5,10,20]:
    fr=px.shift(-h)/px-1
    vals=[]; dates=[]; nvalid=[]
    for dt in f.index:
        x=f.loc[dt]; y=fr.loc[dt]
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
            dates.append(dt); nvalid.append(len(z))
    ic=pd.Series(vals,index=dates).dropna()
    print('H',h,'dates',len(ic),'mean_n',round(float(np.mean(nvalid)),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'min',min(nvalid),'max',max(nvalid))
    # year regimes
    for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-04-03')]:
        q=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
        if len(q): print(' ',lo,hi,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
# coverage and turnover of normalized rank signal
valid=f.notna().sum(axis=1)/len(U)
r=f.rank(axis=1,pct=True); turnover=r.diff().abs().mean(axis=1).mean()
print('coverage',round(valid.mean(),4),'turnover',round(float(turnover),6),'cutoff',px.index.max().date(),'assets',len(frames),'rows',len(px))
# artifact latest signal and all date/symbol values for audit
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20300404_short_reversal_signal.csv',index=False)
print('artifact',len(out))
