import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    p='../persistent/stock_data/'+s+'.csv'
    d=pd.read_csv(p)
    d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
    frames[s]=d['close'].astype(float)
prices=pd.DataFrame(frames).sort_index()
ret5=prices.pct_change(5); ret20=prices.pct_change(20); vol20=prices.pct_change().rolling(20).std()
f=(-ret5 / (vol20*np.sqrt(5))) * (1/(1+np.exp(-ret20/0.05)))
f=f.shift(1)
for h in [5,10,20,40]:
    ic=[]; ns=[]; dates=[]; fr=prices.shift(-h)/prices-1
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
    x=pd.Series(ic,index=dates).dropna(); ir=x.mean()/x.std()
    print(h,'dates',len(x),'avgN',round(np.mean(ns),3),'IC',round(x.mean(),6),'dailyICIR',round(ir,6),'annualICIR',round(ir*np.sqrt(252),6),'hit',round((x>0).mean(),4))
valid=f.notna().sum(axis=1); print('overall dates',len(prices),'assets',len(prices.columns),'coverage',round((valid/len(U)).mean(),4),'active_dates',int((valid>=8).sum()))
fr=prices.shift(-20)/prices-1
for lo in ['2026-01-01','2029-01-01','2031-01-01','2033-01-01']:
    x=[]
    for dt in f.loc[lo:].index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    x=pd.Series(x).dropna(); print('regime',lo,'n',len(x),'IC',round(x.mean(),6),'IR',round(x.mean()/x.std(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340303_conditional_short_reversal_signal.csv',index=False)
