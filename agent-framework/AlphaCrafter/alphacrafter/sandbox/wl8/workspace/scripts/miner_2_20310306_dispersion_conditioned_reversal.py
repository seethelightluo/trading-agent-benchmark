import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fun in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fun(s,days=5000)
   if d is not None and len(d): return d.set_index('date')['close'].astype(float)
  except (FileNotFoundError, Exception): pass
 return pd.Series(dtype=float)
px=pd.DataFrame({s:load(s) for s in U}).sort_index().loc[:'2031-03-05']
ret=px.pct_change(); r5=px.pct_change(5); disp=r5.std(axis=1); med=disp.rolling(252,min_periods=100).median(); strength=(disp/med).clip(.5,2)
f=(-r5).mul(strength,axis=0).replace([np.inf,-np.inf],np.nan); fr=px.shift(-10)/px-1
rows=[]
for i in range(len(px)-10):
 z=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
 if len(z)>=8: rows.append((px.index[i],len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']); y=x.ic.dropna()
def summ(q): return (float(q.mean()),float(q.mean()/q.std(ddof=1)),float((q>0).mean()),len(q)) if len(q)>1 else (np.nan,np.nan,np.nan,len(q))
turn=[]
for i in range(10,len(f),10):
 a=f.iloc[i].rank(pct=True); b=f.iloc[i-10].rank(pct=True); z=pd.concat([a,b],axis=1).dropna(); turn.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('cutoff',px.index[-1].date(),'dates',len(y),'avg_n',round(x.n.mean(),2)); print('daily10d_IC_ICIR_hit',*[round(v,6) if isinstance(v,float) else v for v in summ(y)[:3]]); print('coverage',round(x.n.mean()/15,4),'turnover',round(float(np.mean(turn)),6))
for label,q in [('recent180',y.iloc[-180:]),('recent360',y.iloc[-360:])]: print(label, summ(q)[:3])
print('years'); x['year']=pd.to_datetime(x.date).dt.year
for yr,g in x.groupby('year'): print(int(yr),len(g),round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),6) if len(g)>1 else np.nan)
out=pd.DataFrame({'date':px.index.astype(str)}); [out.__setitem__(s,f[s].values) for s in U]; out.to_csv('scripts/miner_2_20310306_dispersion_conditioned_reversal_signal.csv',index=False)
