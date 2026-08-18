import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

# Volatility-scaled medium trend with drawdown penalty; deliberately one candidate.
acct=get_account_dict(); universe=acct.get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in universe:
    d=get_stock_daily_data(s, days=6000)
    if d is None or len(d)<150: d=get_index_daily_data(s, days=6000)
    if d is not None and len(d)>0: frames[s]=d.set_index('date')['close'].astype(float)
prices=pd.concat(frames,axis=1).sort_index().ffill()
# factor at t uses data through t, then forward returns t+10 / t; signal is lagged in IC pairing
logp=np.log(prices)
r=logp.diff()
ret60=logp-logp.shift(60)
vol30=r.rolling(30).std()*np.sqrt(252)
# current drawdown from 120-day peak, penalty makes trend quality less exposed to damaged assets
peak=prices.rolling(120).max(); dd=prices/peak-1
fac=(ret60/(vol30+1e-8))*(1+dd.clip(-1,0))
fwd=prices.shift(-10)/prices-1
rows=[]
for dt in fac.index:
    x=fac.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
ic=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
# factor turnover: mean rank movement, averaged across consecutive observations
rank=fac.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).mean()
print('candidate=volscaled_trend60_drawdown_penalty')
print('dates=%d instruments=%d coverage=%.4f turnover=%.6f'%(len(ic),len(prices.columns),ic['n'].mean()/len(prices.columns),turnover))
for label,sub in [('full',ic),('120',ic.tail(120)),('252',ic.tail(252)),('756',ic.tail(756)),('1260',ic.tail(1260))]:
    print(label,'IC=%.8f ICIR=%.8f hit=%.4f n=%d'%(sub.ic.mean(),sub.ic.mean()/(sub.ic.std(ddof=1)+1e-12), (sub.ic>0).mean(),len(sub)))
for h in [5,10,20,40]:
    yy=prices.shift(-h)/prices-1; vals=[]
    for dt in fac.index:
        z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
    a=np.array(vals); print('decay',h,'IC=%.8f ICIR=%.8f n=%d'%(np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),len(a)))
# artifacts for provenance
out=fac.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20340120_volscaled_trend60_dd_signal.csv')
ic.to_csv('scripts/miner_1_20340120_volscaled_trend60_dd_ic.csv')
