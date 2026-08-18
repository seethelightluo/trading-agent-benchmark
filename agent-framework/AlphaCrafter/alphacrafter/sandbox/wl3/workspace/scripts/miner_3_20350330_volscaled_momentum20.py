import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<150:d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);px[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill();r=P.pct_change();raw=P.pct_change(20)/r.rolling(20).std();f=raw.sub(raw.mean(axis=1),axis=0).where(P.notna());fw=P.shift(-10)/P-1;rows=[]
for dt in f.index:
 a=f.loc[dt];q=fw.loc[dt];ok=a.notna()&q.notna()
 if ok.sum()>=8:rows.append((dt,a[ok].corr(q[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def st(x):
 x=pd.Series(x).dropna();return (len(x),round(x.mean(),6),round(x.std(),6),round(x.mean()/x.std(),6),round((x>0).mean(),4))
print('range',P.index.min(),P.index.max(),'assets',len(P.columns),'dates',len(z),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/15,4));print('full/recent',st(z.ic),st(z.ic.tail(120)),st(z.ic.tail(252)))
print('blocks',[st(z.ic.iloc[i*len(z)//4:(i+1)*len(z)//4]) for i in range(4)])
print('turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
for h in [1,5,10,20]:
 q=P.shift(-h)/P-1;vals=[]
 for dt in f.index:
  a=f.loc[dt];bb=q.loc[dt];ok=a.notna()&bb.notna()
  if ok.sum()>=8:vals.append(a[ok].corr(bb[ok]))
 print('decay',h,st(vals))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_3_20350330_volscaled_momentum20_signal.csv',index=False)
