import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def main():
 ds={}
 for s in U:
  x=get_stock_daily_data(symbol=s,days=2100)
  if x is None or len(x)<80: continue
  x=x.sort_values('date').reset_index(drop=True); c=x.close.astype(float).values; v=x.volume.astype(float).values
  r=c[1:]/np.maximum(c[:-1],1e-12)-1
  # volume-shock reversal: recent return reversal amplified when participation is unusually high
  l=min(len(r),len(v)-1); dates=x.date.values[1:l+1]
  rr=r[-l:]; vv=v[-l:]
  lv=np.log(np.maximum(vv,1e-12)); z=(lv-pd.Series(lv).rolling(20,min_periods=10).mean().values)/np.maximum(pd.Series(lv).rolling(20,min_periods=10).std().values,1e-8)
  sig=-rr*np.nan_to_num(z,nan=0.)
  ds[s]=pd.DataFrame({'date':dates,'sig':sig,'c':c[1:l+1]})
 rows=[]; turns=[]
 for s,d in ds.items():
  d=d.copy(); d['fwd']=d.c.shift(-1)/d.c-1; d=d.dropna();
  # cross section per date
  rows.append(d[['date','sig','fwd']].assign(s=s))
 all=pd.concat(rows,ignore_index=True)
 ic=all.groupby('date').apply(lambda x:x.sig.corr(x.fwd) if len(x)>=8 and x.sig.nunique()>1 else np.nan).dropna()
 rank=all.assign(r=all.groupby('date').sig.rank(pct=True)).sort_values(['s','date'])
 turns=rank.groupby('s').r.diff().abs().mean()
 print('dates',len(ic),'instruments',all.s.nunique(),'avg_n',all.groupby('date').size().mean(),'coverage',len(all)/(len(ic)*15),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit', (ic>0).mean(),'turnover',turns.mean(),'start',str(ic.index.min()),'end',str(ic.index.max()))
 for h in [3,5,10]:
  z=[]
  for s,d in ds.items():
   q=d.copy(); q['f']=q.c.shift(-h)/q.c-1; z.append(q[['date','sig','f']])
  q=pd.concat(z); a=q.groupby('date').apply(lambda x:x.sig.corr(x.f) if len(x)>=8 else np.nan).dropna(); print('decay',h,a.mean(),len(a))
if __name__=='__main__': main()
