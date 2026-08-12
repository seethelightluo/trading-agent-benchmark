import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; ps={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100: ps[a]=d.set_index('date').close.astype(float)
p=pd.concat(ps,axis=1).sort_index().ffill();r=p.pct_change();v=r.rolling(20).std()
cand=(-p.pct_change(3)/v).stack(); f3=(-p.pct_change(3)).stack(); f5=(-p.pct_change(5)).stack()
q=pd.concat([cand.rename('cand'),f3.rename('f3'),f5.rename('f5')],axis=1).dropna();print('n',len(q),'corr3',q.corr().loc['cand','f3'],'corr5',q.corr().loc['cand','f5'])
