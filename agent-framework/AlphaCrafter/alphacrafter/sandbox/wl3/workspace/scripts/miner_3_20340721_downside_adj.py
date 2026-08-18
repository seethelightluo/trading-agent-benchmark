import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff(); e=r.sub(r.mean(axis=1),axis=0)
# Candidate: downside-risk-adjusted residual momentum. Reward 20d residual trend,
# penalize downside deviation over the same window; lag one completed session.
down=e.where(e<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
trend=e.rolling(20,min_periods=15).sum(); f=(trend/(down*np.sqrt(20)+1e-8)).rank(axis=1,pct=True).sub(.5,axis=0).shift(1)
rows=[]
for h in [1,3,5,10,20]:
 y=lp.shift(-h)-lp
 for dt in f.index:
  a=f.loc[dt]; b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,h,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10,20]:
 q=z[z.h==h].ic.dropna(); print('horizon',h,'dates',len(q),'avgN',z[z.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=z[z.h==10].set_index('date').ic.dropna()
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'dates',len(q),'avg instruments',z[z.h==10].n.mean())
f.to_csv('scripts/miner_3_20340721_downside_adj_signal.csv'); z.to_csv('scripts/miner_3_20340721_downside_adj_ic.csv')
