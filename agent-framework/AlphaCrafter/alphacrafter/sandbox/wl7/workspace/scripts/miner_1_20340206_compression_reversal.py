import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
    d=get_stock_daily_data(s, days=6000)
    if d is not None and len(d):
        z=d[['date','close']].copy(); z['date']=pd.to_datetime(z.date); raw[s]=z.drop_duplicates('date').set_index('date').close
px=pd.DataFrame(raw).sort_index().ffill()
# use only observations through prior completed day relative to supplied current date
px=px.loc[:pd.Timestamp('2034-02-05')]
# Compression-conditioned short-term reversal: oversold recent return is stronger when recent vol compressed vs long vol.
r5=px.pct_change(5); v10=px.pct_change().rolling(10).std(); v60=px.pct_change().rolling(60).std()
# factor higher = expected forward return; reversal sign negative
f=-(r5)*(v60/v10).clip(0.5,3.0)
# activate only compression regime, with neutral otherwise (cross-sectional ranks remain meaningful)
compress=(v10 < v60*0.85)
f=f.where(compress)
f=f.shift(1) # strictly lag signal one completed session
fw=px.shift(-10)/px-1
rows=[]
for dt in f.index:
    a=f.loc[dt]; b=fw.loc[dt]; q=pd.concat([a,b],axis=1).dropna()
    if len(q)>=8:
        ic=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(q)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# exclude terminal dates without forward data already handled
print('range',px.index.min().date(),px.index.max().date(),'dates',len(r),'avgN',r.n.mean(),'coverage',len(r)/(len(px)-10))
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
print('thirds', [round(r.loc[x,'ic'].mean(),6) for x in np.array_split(r.index,3)])
print('recent120',r.tail(120).ic.mean(),r.tail(120).ic.mean()/r.tail(120).ic.std(ddof=1))
# turnover: rank correlation between consecutive factor vectors on common names
rr=[]
for i in range(1,len(f)):
    q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
    if len(q)>=8: rr.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('turnover',np.nanmean(rr),'valid asset coverage',f.notna().sum(axis=1).mean()/len(U))
# signal artifact for provenance
out=f.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20340206_compression_reversal_signal.csv')
