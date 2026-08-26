import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None:return x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
  except:pass
P={s:L(s) for s in U}; px=pd.DataFrame({s:x for s,x in P.items() if x is not None}).sort_index().ffill(limit=3); r=np.log(px).diff()
# Short-term reversal is activated only when cross-asset breadth is extreme,
# reducing exposure to ordinary trending sessions.
rev=-r.rolling(5).sum(); breadth=(r>0).rolling(20).mean().mean(axis=1); disp=r.rolling(20).std().mean(axis=1)
z=(breadth-.5).abs()/(breadth.rolling(120).std().clip(lower=1e-5)); gate=(1-np.exp(-z)).clip(.2,1)
f=rev.div(r.rolling(40).std().clip(lower=1e-5)).mul(gate,axis=0).shift(1); fr=px.shift(-10)/px-1
rows=[]
for d in f.index:
 a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(a)>=8:rows.append((d,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna();m=x.ic.mean();sd=x.ic.std(ddof=1)
print('factor=extreme_breadth_reversal5');print('dates',len(x),'avg_n',x.n.mean(),'coverage',f.notna().sum().sum()/f.size,'IC',m,'ICIR',m/sd*np.sqrt(252),'hit',(x.ic>0).mean());print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2024'),('2025','2029'),('2030','2034'),('2035','2035')]:
 y=x.loc[a:b].ic;print(a,len(y),y.mean(),y.mean()/y.std(ddof=1)*np.sqrt(252) if len(y)>2 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20351210_extreme_breadth_reversal_signal.csv',index=False)
