import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None: F[s]=pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=pd.to_datetime(d.date)).groupby(level=0).last()
px=pd.DataFrame(F).sort_index().ffill(); r=np.log(px).diff()
base=(np.log(px/px.shift(20))/(np.sqrt((r.where(r<0,0)**2).rolling(40,min_periods=20).mean())*np.sqrt(20)+1e-8)).shift(1).clip(-10,10)
ret20=np.log(px/px.shift(20)).shift(1); breadth=(ret20>0).mean(axis=1)
gate=pd.Series(np.where(breadth<1/3,-1.0,1.0),index=px.index); sig=base.mul(gate,axis=0)
out=[]
for dt in px.index:
 z=pd.concat([sig.loc[dt],(px.shift(-10)/px-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: out.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.DataFrame(out,columns=['date','n','ic']).dropna(); m=q.ic.mean(); sd=q.ic.std(ddof=1)
print('candidate breadth_gated_downside_trend20_33')
print('dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.8f ICIR %.8f hit %.4f'%(m,m/sd*np.sqrt(252),(q.ic>0).mean()))
for lab,ss in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]:
 print(lab,len(ss),'IC %.8f ICIR %.8f'%(ss.ic.mean(),ss.ic.mean()/ss.ic.std(ddof=1)*np.sqrt(252)))
for h in [1,5,20,40]:
 oo=[]; fw=px.shift(-h)/px-1
 for dt in px.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: oo.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'IC %.8f'%pd.Series(oo).mean())
rank=sig.rank(axis=1,pct=True); print('coverage',sig.notna().sum().sum()/(len(px)*len(U)),'turnover',rank.diff().abs().mean(axis=1).dropna().mean(),'gate_active',float((breadth<1/3).mean()))
q.to_csv('scripts/miner_2_20300826_breadth_gated_downside_trend20_33_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'asset'}).to_csv('scripts/miner_2_20300826_breadth_gated_downside_trend20_33_signal.csv',index=False)
