import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct=get_account_dict(); uni=acct.get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
uni=[x for x in uni if x in ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']]
cl={}; vol={}
for s in uni:
    d=get_stock_daily_data(s,4000)
    if d is None: d=get_index_daily_data(s,4000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        cl[s]=d['close'].astype(float); vol[s]=d['volume'].astype(float) if 'volume' in d else pd.Series(index=d.index,dtype=float)
prices=pd.DataFrame(cl); volumes=pd.DataFrame(vol).reindex(prices.index)
ret=prices.pct_change()
# volume shock, cross-sectional neutralized 5d reversal, all inputs lagged at signal date
r5=prices.pct_change(5)
vshock=volumes/(volumes.rolling(20,min_periods=10).median())
# cap shock to avoid one-off artifacts; residualize by daily cross-sectional median
raw=-r5*vshock.clip(0.5,3.0)
factor=raw.sub(raw.median(axis=1),axis=0)
rows=[]
for h in [1,5,10,20]:
    f=factor
    fr=prices.shift(-h)/prices-1
    ics=[]; dates=[]; nobs=[]; turnovers=[]
    for dt in f.index:
        x=f.loc[dt]; y=fr.loc[dt]
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); nobs.append(len(z))
        # turnover of cross-sectional ranks vs prior valid date
    ic=pd.Series(ics,index=dates).dropna()
    print('H',h,'IC %.6f ICIR %.4f hit %.3f dates %d avgN %.2f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),len(ic),np.mean(nobs)))
# rank turnover on consecutive dates where both have >=8
r=factor.rank(axis=1,pct=True); changes=[]
for i in range(1,len(r)):
    z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
    if len(z)>=8: changes.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('universe',len(prices.columns),'dates',len(prices),'coverage',factor.notna().mean().mean(),'turnover_proxy',np.mean(changes),'last',prices.index.max())
# regime 10d
fr=prices.shift(-10)/prices-1; icrows=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: icrows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.Series(dict(icrows));
for lo,hi in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034')]:
 q=a[(a.index>=lo)&(a.index<=hi)]; print('REG',lo,hi,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
# signal artifact for audit
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20341127_volshock_reversal_signal.csv',index=False)
