import os, pandas as pd, numpy as np
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'
frames={}
for s in U:
    p=os.path.join(root,s+'.csv')
    d=pd.read_csv(p)
    d['date']=pd.to_datetime(d['date'])
    d=d.sort_values('date').set_index('date')
    frames[s]=d
# lagged 20d return divided by lagged 20d realized vol, with volatility floor; signal available through prior close
prices=pd.concat({s:frames[s]['close'] for s in U},axis=1).sort_index()
r=prices.pct_change()
ret20=prices.shift(1)/prices.shift(21)-1
vol20=r.rolling(20).std().shift(1)*np.sqrt(20)
sig=ret20/(vol20+1e-8)
fwd=prices.shift(-1)/prices-1
rows=[]
for dt in prices.index:
    if dt>pd.Timestamp('2028-05-31'): continue
    x=sig.loc[dt]; y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'rows',a.n.sum(),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
for name,sub in [('all',a),('2020-22',a.loc['2020':'2022']),('2023-25',a.loc['2023':'2025']),('2026',a.loc['2026']),('2027',a.loc['2027']),('2028',a.loc['2028']),('last180',a.tail(180))]:
    if len(sub): print(name,'n',len(sub),'IC %.6f ICIR %.6f hit %.3f'%(sub.ic.mean(),sub.ic.mean()/(sub.ic.std(ddof=1)+1e-12)*np.sqrt(len(sub)),(sub.ic>0).mean()))
# signal turnover via rank changes on consecutive valid dates
ranks=sig.rank(axis=1,pct=True)
t=(ranks.diff().abs().mean(axis=1)).dropna().mean()
print('turnover_proxy',t)
for h in [1,3,5,10]:
    fw=prices.shift(-h)/prices-1; rr=[]
    for dt in prices.index:
      if dt>pd.Timestamp('2028-05-31'): continue
      z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
      if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    print('horizon',h,'dates',len(rr),'IC',np.nanmean(rr),'ICIR',np.nanmean(rr)/(np.nanstd(rr,ddof=1)+1e-12)*np.sqrt(len(rr)))
