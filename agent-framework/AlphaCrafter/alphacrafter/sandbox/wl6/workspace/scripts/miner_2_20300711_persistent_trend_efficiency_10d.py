import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,3000) for s in U}
px=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index().ffill()
# signal uses only close through date t: medium trend efficiency times directional persistence
r=px.pct_change(); mom=px.pct_change(40); vol=r.rolling(40).std()*np.sqrt(252)
# fraction of positive daily returns, centered so persistent upward trends score higher
persistence=r.gt(0).rolling(40).mean()-0.5
f=(mom/(vol+1e-8))*(1+0.8*persistence)
print('data_range',px.index.min(),px.index.max(),'rows',len(px))
for h in [5,10,20]:
    vals=[]; ns=[]; turns=[]
    for i in range(40,len(px)-h):
        z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
            if i>40: turns.append((f.iloc[i].rank(pct=True)-f.iloc[i-1].rank(pct=True)).abs().mean())
    a=np.asarray(vals); print('horizon',h,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,5),'IC',round(a.mean(),8),'ICIR',round(a.mean()/(a.std(ddof=1)/np.sqrt(len(a))),5),'hit',round(np.mean(a>0),5),'turnover',round(np.mean(turns),6))
ser=[]
for i in range(40,len(px)-10):
    z=pd.concat([f.iloc[i],px.iloc[i+10]/px.iloc[i]-1],axis=1).dropna()
    if len(z)>=8: ser.append((px.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
for yr in sorted(set(d.year for d,v in ser)):
    q=[v for d,v in ser if d.year==yr]; print('regime',yr,'dates',len(q),'mean_ic',round(float(np.mean(q)),6))
