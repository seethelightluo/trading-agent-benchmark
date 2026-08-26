import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# DXY is observation-only, never tradable
px={}
for s in U:
    d=get_stock_daily_data(s, days=6000)
    if d is None or len(d)<300: d=get_index_daily_data(s, days=6000)
    if d is not None: px[s]=d.set_index('date')['close'].astype(float)
dxy=get_index_daily_data('DXY', days=6000)
if dxy is None: dxy=get_stock_daily_data('DXY', days=6000)
dxy=dxy.set_index('date')['close'].astype(float)
prices=pd.DataFrame(px).sort_index(); ret=np.log(prices).diff(); dr=np.log(dxy).diff().reindex(prices.index)
# lagged signal: 20d risk-adjusted trend, with a DXY-regime overlay. DXY weakening historically supports risk assets; invert overlay for assets via common multiplier.
base=ret.rolling(20).sum()/ret.rolling(40).std().replace(0,np.nan)
dxy5=dr.rolling(5).sum(); dxy_level=dxy.rolling(252).rank(pct=True)
# multiplier only when DXY is unusually strong and turning down, emphasizing trend/rebound after macro pressure eases
mult=1+0.50*((dxy_level.shift(1)>0.70)&(dxy5.shift(1)<0)).astype(float)
f=base.shift(1).mul(mult,axis=0)
# forward 10d returns
fr=prices.shift(-10)/prices-1
rows=[]; sig=[]
for dt in f.index:
    x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ic=x[ok].corr(y[ok],method='spearman'); rows.append((dt,ic,ok.sum())); sig.append((dt,*x.tolist()))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); vals=ic.ic.dropna()
print('factor=dxy_conditioned_trend dates',len(ic),'avgN',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC10',vals.mean(),'ICIR_daily',vals.mean()/vals.std(),'hit',(vals>0).mean(),'recent365',vals.tail(365).mean()/vals.tail(365).std())
for h in [1,5,10,20]:
    yy=prices.shift(-h)/prices-1; rr=[]
    for dt in f.index:
        x=f.loc[dt]; y=yy.loc[dt]; ok=x.notna()&y.notna()
        if ok.sum()>=8: rr.append(x[ok].corr(y[ok],method='spearman'))
    print('decay',h,np.nanmean(rr))
# turnover proxy rank signal changes
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
ic.to_csv('scripts/miner_2_20341221_dxy_conditioned_trend_ic.csv')
pd.DataFrame(sig,columns=['date']+U).set_index('date').to_csv('scripts/miner_2_20341221_dxy_conditioned_trend_signal.csv')
