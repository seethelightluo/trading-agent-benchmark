import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None: px[s]=d.set_index('date')['close'].astype(float)
D=pd.DataFrame(px).sort_index().loc[:pd.Timestamp('2035-02-17')]
rows=[];out=[]
for i in range(30,len(D)-10):
 f=[];r=[]
 for s in U:
  x=D[s].iloc[:i+1]
  if len(x)<21 or pd.isna(x.iloc[-1]): continue
  ret=x.pct_change().iloc[-20:]; sig=-ret.std()
  fw=D[s].iloc[i+10]/D[s].iloc[i]-1
  if np.isfinite(sig) and np.isfinite(fw): f.append(sig);r.append(fw);out.append({'date':D.index[i],'symbol':s,'signal':sig})
 if len(f)>=8: rows.append((D.index[i],spearmanr(f,r).statistic,len(f)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034'),('2035','2035')]:
 q=r.loc[a:b].ic;print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('turnover',pd.DataFrame(out).pivot(index='date',columns='symbol',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
pd.DataFrame(out).to_csv('scripts/miner_2_20350219_lowvol20_signal.csv',index=False)
