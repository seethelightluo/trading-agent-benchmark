import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d): D[s]=d.set_index('date')
px=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); hi=pd.DataFrame({s:d.high.astype(float) for s,d in D.items()}).reindex(px.index); lo=pd.DataFrame({s:d.low.astype(float) for s,d in D.items()}).reindex(px.index)
r=np.log(px).diff(); rv=r.rolling(20).std(); tr=(hi-lo)/px
# Activate reversal after a cross-sectionally abnormal range day: range above
# the same asset's rolling 70th percentile, with a 5-day persistence window.
exp=(tr.shift(1)>tr.shift(1).rolling(20,min_periods=20).quantile(.70))
base=-(r.rolling(5).sum()/rv).shift(1)
f=base.where(exp.rolling(5,min_periods=1).max().astype(bool))
f=f.sub(f.median(axis=1),axis=0)
print('dates',len(px),'instruments',len(D),'raw_valid',base.notna().sum(axis=1).mean(),'active',f.notna().sum(axis=1).mean())
for h in [1,3,5,10]:
 rows=[]
 for i in range(len(px)-h):
  q=pd.DataFrame({'a':f.iloc[i],'b':np.log(px.iloc[i+h]/px.iloc[i])}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1: rows.append((px.index[i],len(q),q.a.corr(q.b,method='spearman')))
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
 print('H',h,'obs',len(x),'avgN',round(x.n.mean(),2),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lab,q in [('2020_22',x.loc['2020':'2022']),('2023_25',x.loc['2023':'2025']),('2026_27',x.loc['2026':'2027']),('2028_29',x.loc['2028':'2029']),('recent250',x.tail(250))]: print(lab,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(),6))
print('coverage',f.notna().sum().sum()/(len(f)*len(D)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300207_expansion_reversal_signal.csv',index=False)
