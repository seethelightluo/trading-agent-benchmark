import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def one(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 if d is None or len(d)<40:return None
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index()
 rv=d.close.pct_change().rolling(20,min_periods=15).std()
 # Signal is lagged one completed session: f[t] uses data through t-1.
 f=(-(d.close.pct_change(1))).shift(1)
 r=d.close.shift(-1)/d.close-1
 return pd.DataFrame({'symbol':s,'f':f,'r':r}).replace([np.inf,-np.inf],np.nan).dropna().reset_index()
def main():
 qs=[one(s) for s in U];qs=[x for x in qs if x is not None];x=pd.concat(qs,ignore_index=True)
 vals=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: vals.append((dt,g.f.corr(g.r,method='spearman'),len(g)))
 z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date');z['ic']=z.ic.astype(float)
 for name,a in [('all',z),('2020_22',z.loc[:'2022-12-31']),('2023_25',z.loc['2023-01-01':'2025-12-31']),('2026+',z.loc['2026-01-01':]),('recent252',z.tail(252))]:
  print(name,'dates',len(a),'meanIC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(),5),'hit',round((a.ic>0).mean(),4))
 print('instruments',x.symbol.nunique(),'rows',len(x),'coverage_rows_per_possible',round(len(x)/(len(z)*15),4),'avg_n',round(z.n.mean(),2))
 # signal artifact for deterministic audit
 x[['date','symbol','f']].to_csv('scripts/miner_1_20281214_raw_reversal1_signal.csv',index=False)
if __name__=='__main__':main()
