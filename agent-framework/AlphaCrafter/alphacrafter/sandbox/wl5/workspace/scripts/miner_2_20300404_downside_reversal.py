import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 f='../persistent/index_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): D[s]=pd.read_csv(f,parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();mom=p/p.shift(10)-1
# Downside semideviation, floored by total volatility to avoid unstable zero denominators.
down=r.where(r<0).rolling(30,min_periods=10).std()*np.sqrt(252); total=r.rolling(30,min_periods=10).std()*np.sqrt(252)
den=down.fillna(total).clip(lower=0.05)
sig=-(mom/den)
h=10;fwd=p.shift(-h)/p-1;vals=[];ds=[];ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c):vals.append(c);ds.append(dt);ns.append(len(z))
q=pd.Series(vals,index=ds);print('horizon',h,'dates',len(q),'avg_instruments %.2f'%np.mean(ns),'IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
for label,lo,hi in [('2020-24','2020','2025'),('2025-27','2025','2028'),('2028-30','2028','2030-04-05')]:
 t=q[(q.index>=lo)&(q.index<hi)];print(label,'dates',len(t),'IC %.8f ICIR %.8f hit %.4f'%(t.size and t.mean(),t.size and t.mean()/t.std(),(t>0).mean()))
print('coverage %.6f turnover %.6f instruments %d rows %d'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean(),len(D),int(sig.stack().size)))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20300404_downside_reversal_signal.csv',index=False);print('artifact',len(out))
