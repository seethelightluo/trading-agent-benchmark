import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
def main():
 ds={}
 for s in U:
  x=get_stock_daily_data(symbol=s,days=3000)
  if x is not None and len(x)>120:
   x=x.sort_values('date').reset_index(drop=True); ds[s]=x
 dates=ds['SPX'].date.tolist(); rows=[]
 for j in range(65,len(dates)-10):
  d=dates[j]; vals=[]; fw=[]
  for s,x in ds.items():
   k=x.date.searchsorted(d,side='right')-1; kf=x.date.searchsorted(dates[j+10],side='left')
   if k>=60 and kf<len(x):
    c=x.close.astype(float); vol=x.volume.astype(float).replace(0,np.nan)
    rv=np.std(c.pct_change().iloc[k-19:k+1]); av=np.log(vol.iloc[k-4:k+1].mean()/vol.iloc[k-59:k+1].mean())
    vals.append((c.iloc[k]/c.iloc[k-10]-1)*av/max(rv,.003)); fw.append(c.iloc[kf]/c.iloc[k]-1)
  if len(vals)>=8 and np.std(vals)>0 and np.std(fw)>0: rows.append((d,np.corrcoef(vals,fw)[0,1],len(vals)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
 print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',ic,'ICIR',ir,'hit',np.mean(q.ic>0))
 print('annual',q.assign(year=pd.to_datetime(q.date).dt.year).groupby('year').ic.mean().to_string())
 print('last',q.tail(5).to_string(index=False))
if __name__=='__main__': main()
