import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is None or len(d)<150: d=get_index_daily_data(s, days=4000)
    if d is not None and len(d)>150:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').drop_duplicates('date'); frames[s]=d.set_index('date')
close=pd.DataFrame({s:d['close'] for s,d in frames.items()}); vol=pd.DataFrame({s:d['volume'] for s,d in frames.items()}).replace(0,np.nan)
ret=close.pct_change(); rv=ret.rolling(20).std(); vshock=vol/vol.rolling(60).median()-1
# Contrarian response to a sharp, unusually liquid 5-day move, scaled by risk.
factor=-(close.pct_change(5)/ (rv*np.sqrt(5))).shift(1) * (1+vshock.shift(1).clip(lower=0,upper=3))
# signal clipped cross-sectionally only via ranks is unnecessary for IC
rows=[]
for dt in close.index:
    if dt not in factor.index: continue
    f=factor.loc[dt]; fr=close.shift(-10).loc[dt]/close.loc[dt]-1
    z=pd.concat([f,fr],axis=1).dropna()
    if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [5,10,20,30]:
    fr=close.shift(-h)/close-1; vals=[]
    for dt in factor.index:
        z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
    a=pd.Series(vals).dropna(); print('H',h,'dates',len(a),'N',round(float(np.mean([n for _,_,n in rows])),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),4),'hit',round((a>0).mean(),4))
# recent regimes
for label, q in [('recent365',x.tail(365)),('recent120',x.tail(120)),('early',x.head(365))]: print(label,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),4),'hit',round((q.ic>0).mean(),4))
# coverage and turnover on date-aligned signal
print('assets',len(frames),'coverage',round(factor.notna().sum(axis=1).div(len(frames)).mean(),4),'turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
# artifact for deterministic audit
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_2_20330331_liquidity_shock_reversal_signal.csv',index=False)
