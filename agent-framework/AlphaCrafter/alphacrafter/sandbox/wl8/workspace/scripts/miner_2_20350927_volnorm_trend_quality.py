import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def fetch(s):
    try: return get_index_daily_data(s,days=6000)
    except FileNotFoundError: return get_stock_daily_data(s,days=6000)
D={s:fetch(s) for s in U}
px={s:(d.set_index('date')['close'].astype(float) if d is not None else pd.Series(dtype=float)) for s,d in D.items()}
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
ret=P.pct_change(20); vol=R.rolling(20).std()*np.sqrt(20); eff=ret.abs()/(R.abs().rolling(20).sum()+1e-12)
f=(ret/(vol+1e-12))*eff
rows=[]
for i in range(1,len(P)-10):
 sig=f.iloc[i-1]; fr=P.iloc[i+9]/P.iloc[i]-1; z=pd.concat([sig.rename('x'),fr.rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[i],len(z),z.x.corr(z.y,method='spearman')))
q=pd.DataFrame(rows,columns=['date','n','ic']).dropna()
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,20]:
 rr=[]
 for i in range(1,len(P)-h):
  z=pd.concat([f.iloc[i-1].rename('x'),(P.iloc[i+h-1]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rr.append(z.x.corr(z.y,method='spearman'))
 print('decay',h,np.nanmean(rr),len(rr))
for n in [365,750,1260]: print('recent',n,q.tail(n).ic.mean(),q.tail(n).ic.mean()/q.tail(n).ic.std())
q.to_csv('scripts/miner_2_20350927_volnorm_trend_quality_ic.csv',index=False); f.tail(3000).to_csv('scripts/miner_2_20350927_volnorm_trend_quality_signal.csv')
