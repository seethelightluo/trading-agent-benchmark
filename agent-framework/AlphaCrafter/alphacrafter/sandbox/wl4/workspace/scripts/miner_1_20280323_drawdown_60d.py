import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; frames={}
for s in U:
 try:d=get_index_daily_data(s,days=5000)
 except:d=None
 if d is None:
  try:d=get_stock_daily_data(s,days=5000)
  except:d=None
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);frames[s]=d.sort_values('date').drop_duplicates('date').set_index('date')
rows=[]
for s,d in frames.items():
 for dt in d.index:
  h=d.loc[:dt]
  if len(h)<65:continue
  px=h.close.iloc[-1]; peak=h.close.tail(60).max(); f=-float(px/peak-1) # negative drawdown: closer to high is higher
  fut=d.loc[d.index>dt,'close']
  if len(fut)>=10:rows.append((dt,s,f,float(fut.iloc[0]/px-1),float(fut.iloc[9]/px-1)))
for k,col in [(3,'fwd1'),(4,'fwd10')]:
 z=pd.DataFrame(rows,columns=['date','symbol','f','fwd1','fwd10']);a=[]
 for _,g in z.groupby('date'):
  if len(g)>=8:a.append(spearmanr(g.f,g[col]).statistic)
 a=np.array(a);print(col,'dates',len(a),'avg_n',z.groupby('date').size().mean(),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
print('symbols',len(frames))
