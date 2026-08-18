import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acct=get_account_dict(); universe=acct.get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
universe=[x if isinstance(x,str) else x.get('symbol') for x in universe]
prices={}
for s in universe:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        d=d[['date','close']].drop_duplicates('date').set_index('date')['close'].astype(float)
        prices[s]=d
px=pd.DataFrame(prices).sort_index().ffill()
# factor: lagged residualized trend, conditioned on broad risk regime. Positive trend is used only
# when market trend is positive; in weak regime, defensive assets' relative trend remains selected.
ret=px.pct_change(); mret=ret.mean(axis=1)
asset20=px.pct_change(20).shift(1); market20=mret.rolling(20).sum().shift(1)
vol=ret.rolling(20).std().shift(1)*np.sqrt(252)
res=(asset20.sub(market20,axis=0)).div(vol.replace(0,np.nan))
# smooth regime: continuous positive breadth/risk score, bounded and fully lagged
breadth=(ret.rolling(20).sum().shift(1)>0).mean(axis=1)
reg=(0.5+0.5*((breadth-0.5)*2)).clip(0,1) # 0..1, doesn't zero factor; market breadth conditioning
f=res.mul(reg,axis=0)
# 10-day forward return
fwd=px.shift(-10).div(px)-1
rows=[]
for dt in f.index:
    a=f.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('universe',len(universe),'available',len(prices),'dates',len(r),'avg_n',round(r.n.mean(),3))
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean(), r.n.mean()/len(prices)))
for start,end in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-12-31')]:
 q=r.loc[start:end]; print(start,'n',len(q),'ic',round(q.ic.mean(),6),'ir',round(q.ic.mean()/q.ic.std(ddof=1),6))
# signal artifact
out=f.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20330916_vol_conditioned_residual_signal.csv')
# turnover rank changes
rank=f.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)>0.05).mean()
print('turnover_event_rate',round(float(turn),4))
for h in [1,3,5,10]:
 fw=px.shift(-h).div(px)-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,'dates',len(vals),'ic',round(float(np.nanmean(vals)),6),'ir',round(float(np.nanmean(vals)/np.nanstd(vals,ddof=1)),6))
