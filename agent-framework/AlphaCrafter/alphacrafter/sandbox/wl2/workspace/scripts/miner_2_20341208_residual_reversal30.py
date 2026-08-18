import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<100:d=get_index_daily_data(s,3000)
 D[s]=d
p=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index(); r=np.log(p).diff()
# 30d cross-asset residual reversal, volatility scaled, 20d forward horizon
cum=r.rolling(30,min_periods=25).sum(); common=r.mean(axis=1); resid=r.sub(common,axis=0); rv=resid.rolling(60,min_periods=40).std(); rows=[]
for i in range(80,len(p)-20):
 f=-(resid.iloc[i-29:i+1].sum())/(rv.iloc[i]+1e-8)
 fw=np.log(p.iloc[i+20]/p.iloc[i]); z=pd.concat([f.rename('f'),fw.rename('fw')],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],z.f.corr(z.fw),len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(o),'mean_n',o.n.mean(),'coverage',o.n.mean()/15,'meanIC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(),'hit',(o.ic>0).mean(),'turnover_proxy',o.ic.diff().abs().mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2032-12-31'),('2033','2034-12-31')]:
 x=o.loc[a:b,'ic']; print(a,b,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std() if x.std()>0 else np.nan)
for h in [5,10,20,40]:
 rr=[]
 for i in range(80,len(p)-h):
  f=-(resid.iloc[i-29:i+1].sum())/(rv.iloc[i]+1e-8); fw=np.log(p.iloc[i+h]/p.iloc[i]); z=pd.concat([f.rename('f'),fw.rename('fw')],axis=1).dropna()
  if len(z)>=8: rr.append(z.f.corr(z.fw))
 print('decay',h,np.nanmean(rr),len(rr))
o.to_csv('../persistent/miner_2_20341208_residual_reversal30_ic.csv')
# Persist reproducible signal artifact: date, symbol, factor value
ss=[]
for i in range(80,len(p)-20):
 f=-(resid.iloc[i-29:i+1].sum())/(rv.iloc[i]+1e-8)
 for sym,val in f.dropna().items(): ss.append({'date':p.index[i], 'symbol':sym, 'signal':float(val)})
pd.DataFrame(ss).to_csv('../persistent/miner_2_20341208_residual_reversal30_signal.csv',index=False)
