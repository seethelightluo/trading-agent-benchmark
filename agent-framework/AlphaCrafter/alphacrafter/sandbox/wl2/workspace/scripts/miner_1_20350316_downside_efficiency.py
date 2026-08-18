import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
acct=get_account_dict(); syms=acct.get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 if d is not None and len(d): prices[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(prices).sort_index(); r=np.log(p).diff(); mom=r.rolling(30).sum(); down=r.where(r<0,0).pow(2).rolling(30).mean().pow(.5); f=mom/down; f=f.sub(f.median(axis=1),axis=0)
rows=[]; dates=[]; ns=[]
for i in range(150,len(p)-10):
 z=pd.concat([f.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(p.index[i]); ns.append(len(z))
ic=pd.Series(rows,index=dates).dropna(); print('dates',len(ic),'universe',len(syms),'mean_n',np.mean(ns)); print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for h in [1,3,5,10,20]:
 vals=[]
 for i in range(150,len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(vals),len(vals))
for a,b in [('2020','2024'),('2025','2029'),('2030','2035')]:
 q=ic[(ic.index>=a)&(ic.index<=b+'-12-31')]; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean().mean())
out=f.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('../persistent/miner_1_20350316_downside_efficiency_signal.csv'); pd.DataFrame({'date':ic.index.strftime('%Y-%m-%d'),'ic':ic.values}).to_csv('../persistent/miner_1_20350316_downside_efficiency_ic.csv',index=False)
