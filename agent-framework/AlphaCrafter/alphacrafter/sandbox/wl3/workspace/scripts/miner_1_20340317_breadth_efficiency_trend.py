import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(s):
    d=get_stock_daily_data(s, days=6000)
    if d is None or len(d)<300: d=get_index_daily_data(s, days=6000)
    return d[['date','close','volume']].copy() if d is not None else None
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
px=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill()
vol=pd.concat({s:d.set_index('date').volume for s,d in D.items()},axis=1).sort_index().reindex(px.index).ffill()
r=np.log(px).diff()
# candidate: 30d residual trend, scaled by path efficiency and idio volatility; breadth gate based on lagged 20d breadth
mkt=r.mean(axis=1)
res=r.sub(mkt,axis=0)
raw=res.rolling(30).sum()
idvol=res.rolling(45).std()*np.sqrt(252)
eff=(res.rolling(30).sum().abs()/(res.abs().rolling(30).sum()+1e-12)).clip(0,1)
breadth=(r.rolling(20).sum()>0).mean(axis=1)
gate=(0.5+0.5*breadth).rolling(10).mean()
f=(raw/(idvol+1e-12)*eff).shift(1).mul(gate.shift(1),axis=0)
fr=px.shift(-10)/px-1
rows=[]; ics=[]
for dt in px.index:
    x=f.loc[dt]; y=fr.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        if np.isfinite(ic): ics.append((dt,ic,len(z)))
q=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date')
mean=q.ic.mean(); sd=q.ic.std(ddof=1); icir=mean/sd*np.sqrt(252/10) if sd else np.nan
turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('dates',len(q),'avg_n',q.n.mean(),'coverage',len(q)/len(px.index),'IC10',mean,'ICIR10',icir,'hit', (q.ic>0).mean(),'turn',turn)
for n in [120,252,756,1260]:
 z=q.tail(n); print('recent',n,z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252/10) if len(z)>2 else np.nan)
print('decay')
for h in [5,10,20]:
 yy=px.shift(-h)/px-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(h,np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(252/h))
# artifact
f.to_csv('scripts/miner_1_20340317_breadth_efficiency_trend_signal.csv',index_label='date')
q.to_csv('scripts/miner_1_20340317_breadth_efficiency_trend_ic.csv')
