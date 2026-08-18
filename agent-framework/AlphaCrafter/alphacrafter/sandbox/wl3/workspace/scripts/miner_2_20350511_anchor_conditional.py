import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
px=pd.DataFrame({s:d['close'] for s,d in D.items()}).sort_index().ffill(); r=px.pct_change(); quality=px.pct_change(60)/(r.rolling(40).std()*np.sqrt(252))
eq=U[:8]; defens=['XAU','US10Y','CN10Y']; reg=(px[defens].pct_change(20).mean(axis=1)-px[eq].pct_change(20).mean(axis=1)).shift(1)
f=quality.sub(quality.median(axis=1),axis=0); sign=pd.Series(np.where(reg<=0,1.,-1.),index=px.index); signal=f.mul(sign,axis=0)
for name,x in [('anchor_conditional',signal),('plain',f)]:
 ic=[]; n=[]; turns=[]
 for i in range(len(x.index)-10):
  z=x.iloc[i].dropna(); y=(px.iloc[i+10]/px.iloc[i]-1).reindex(z.index).dropna(); z=z.reindex(y.index)
  if len(z)>=8:
   ic.append(z.corr(y,method='spearman')); n.append(len(z))
   if i>0: turns.append((x.iloc[i].rank()-x.iloc[i-1].rank()).abs().mean()/len(z))
 a=pd.Series(ic).dropna(); print(name,'dates',len(a),'avgN',np.mean(n),'IC10',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turn',np.mean(turns),'recent120',a.tail(120).mean(),'recent252',a.tail(252).mean())
signal.to_csv('scripts/miner_2_20350511_anchor_conditional_signal.csv')
