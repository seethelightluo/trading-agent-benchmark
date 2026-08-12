import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={};O={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  q=d.set_index('date'); C[s]=q.close.astype(float); O[s]=q.open.astype(float)
px=pd.DataFrame(C).sort_index(); op=pd.DataFrame(O).reindex(px.index)
gap=np.log(op/px.shift(1)); vol=np.log(px).diff().rolling(20).std()*np.sqrt(20)
f=(-gap.rolling(3,min_periods=2).mean()/(vol+1e-9)).shift(1); f=f.sub(f.median(axis=1),axis=0)
print('dates',len(px),'instruments',len(C))
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index[:-h]):
  a=f.iloc[i]; b=np.log(px.iloc[i+h]/px.iloc[i]); q=pd.DataFrame({'a':a,'b':b}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1: rows.append((dt,len(q),q.a.corr(q.b,method='spearman')))
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
 print('H',h,'obs',len(x),'avgN %.2f'%x.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
 for lab,q in [('2020_22',x.loc['2020':'2022']),('2023_25',x.loc['2023':'2025']),('2026_27',x.loc['2026':'2027']),('2028_29',x.loc['2028':'2029']),('recent250',x.tail(250))]: print(lab,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std()))
rr=f.rank(axis=1,pct=True); print('coverage %.4f turnover %.4f'%(f.notna().sum().sum()/(len(f)*len(C)),rr.diff().abs().mean(axis=1).mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300110_overnight_gap_signal.csv',index=False)
