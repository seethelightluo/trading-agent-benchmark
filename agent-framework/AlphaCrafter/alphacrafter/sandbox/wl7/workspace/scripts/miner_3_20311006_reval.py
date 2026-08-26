import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); vol=r.rolling(40,min_periods=30).std()
def z(x): return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1).replace(0,np.nan),axis=0)
ret20=P.pct_change(20); ret10=P.pct_change(10); prior10=ret10.shift(10)
rev=-ret20.sub(ret20.mean(axis=1),axis=0)
acc=(ret10-prior10)/vol
sig=(.60*z(rev)+.40*z(acc)).shift(1)
def ev(h):
 y=P.shift(-h)/P-1; out=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8: out.append((dt,sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'),int(v.sum())))
 a=pd.Series([x[1] for x in out]); return a,out
for h in [1,5,10]:
 a,o=ev(h); print(f'h={h} dates={len(a)} avg_n={np.mean([x[2] for x in o]):.2f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1):.8f} hit={(a>0).mean():.5f}')
a,o=ev(1); print('history_dates',len(P),'assets',len(P.columns),'coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
print('regimes',*[round(a.iloc[i:j].mean(),8) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(o,columns=['date','ic','n']).to_csv('scripts/miner_3_20311006_reval_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20311006_reval_signal.csv',index=False)
