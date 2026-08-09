import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2026-07-15'").set_index('date').sort_index() for s in U}
rows=[]
for s,x in D.items():
    intra=-(x.close/x.open-1)
    rev5=-(x.close/x.close.shift(5)-1)
    mom20=x.close/x.close.shift(20)-1
    rng=x.high-x.low
    clv=-(2*(x.close-x.low)/rng-1).where(rng!=0)
    # peer-median leadlag: own 5d return minus cross-sectional median on same date
    r5=x.close/x.close.shift(5)-1
    rows.append(pd.DataFrame({'date':x.index,'intra':intra,'rev5':rev5,'mom20':mom20,'clv':clv,'r5':r5,'s':s}))
a=pd.concat(rows,ignore_index=True).dropna(subset=['intra','rev5','mom20','clv'])
print('pooled_corr',a[['intra','rev5','mom20','clv']].corr().round(6).to_dict()['intra'])
# correlation after date demeaning, more relevant cross-sectional exposure overlap
for c in ['rev5','mom20','clv']:
    q=a[['date','intra',c]].copy(); q['i']=q.intra-q.groupby('date').intra.transform('mean'); q['j']=q[c]-q.groupby('date')[c].transform('mean'); print('demeaned',c,q.i.corr(q.j))
print('n',len(a),'dates',a.date.nunique())
