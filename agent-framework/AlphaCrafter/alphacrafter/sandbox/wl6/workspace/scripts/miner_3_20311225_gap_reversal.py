import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=3000)
    if x is None or len(x)<100: continue
    x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date')
    x['r']=x.close.pct_change(); x['gap']=x.open/x.close.shift(1)-1
    x['vol']=x.r.rolling(20).std(); x['fwd10']=x.close.shift(-10)/x.close-1
    D[s]=pd.DataFrame({'sig':-x.gap/x.vol.replace(0,np.nan),'fwd':x.fwd10})
rows=[]
for dt in sorted(set().union(*[set(x.index) for x in D.values()])):
    z=pd.DataFrame({s:{k:v.get(dt,np.nan) for k,v in x.items()} for s,x in D.items()}).T.dropna()
    if len(z)>=8:
        ic=z.sig.corr(z.fwd,method='spearman')
        if np.isfinite(ic): rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'assets',len(D))
print('IC %.6f ICIR %.6f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
for label,sub in r.groupby(pd.cut(r.index.year,[2019,2022,2025,2028,2032])):
 print(label,'n',len(sub),'IC %.6f ICIR %.6f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1)))
# rank turnover
all_sig=pd.DataFrame({s:x.sig for s,x in D.items()}); print('turnover',all_sig.rank(axis=1,pct=True).diff().abs().mean().mean())
print('period',r.index.min(),r.index.max())
