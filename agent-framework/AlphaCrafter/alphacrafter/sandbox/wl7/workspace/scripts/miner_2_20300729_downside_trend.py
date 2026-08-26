import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=3000)
 if d is not None:F[s]=pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=pd.to_datetime(d.date)).groupby(level=0).last()
px=pd.DataFrame(F).sort_index().ffill(); r=np.log(px).diff(); sig=(np.log(px/px.shift(20))/(np.sqrt((r.where(r<0,0)**2).rolling(40,min_periods=20).mean())*np.sqrt(20)+1e-8)).shift(1).clip(-10,10)
def calc(h):
 fw=px.shift(-h)/px-1;out=[]
 for dt in px.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:out.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 return pd.DataFrame(out,columns=['date','n','ic']).dropna()
for h in [1,5,10,20,40]:
 q=calc(h);m=q.ic.mean();print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.8f ICIR %.8f hit %.4f'%(m,m/q.ic.std(ddof=1)*np.sqrt(252),(q.ic>0).mean()))
 if h==10:
  for lab,ss in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]:print(lab,len(ss),'IC %.8f ICIR %.8f'%(ss.ic.mean(),ss.ic.mean()/ss.ic.std(ddof=1)*np.sqrt(252)))
  q.to_csv('scripts/miner_2_20300729_downside_trend_ic.csv',index=False)
rank=sig.rank(axis=1,pct=True);print('assets',len(F),'dates',len(px),'coverage',sig.notna().sum().sum()/(len(px)*len(U)),'turnover',rank.diff().abs().mean(axis=1).dropna().mean())
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'asset'}).to_csv('scripts/miner_2_20300729_downside_trend_signal.csv',index=False)
