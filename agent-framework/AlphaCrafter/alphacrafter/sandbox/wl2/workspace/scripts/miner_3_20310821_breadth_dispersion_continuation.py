import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; idx='../persistent/index_data'; end='2031-08-21'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); R=P.pct_change();
# Continuation in broad risk-on conditions: 20d residual momentum, activated by improving breadth and compressed dispersion.
r20=P.pct_change(20); resid=r20.sub(r20.median(axis=1),axis=0)
breadth=(R>0).sum(axis=1)/len(U); disp=R.std(axis=1)
active=(breadth.rolling(5).mean()>0.60)&(disp<disp.rolling(252,min_periods=126).median())&(breadth.diff(5)>0)
f=resid.where(active, np.nan); y=P.pct_change().shift(-1)
rows=[]; sig=[]
for d in P.index:
 x=f.loc[d]; z=y.loc[d]; ok=x.notna()&z.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],z[ok]).statistic
  if np.isfinite(ic): rows.append((d,ic,ok.sum()))
 for s in U: sig.append((d,s,f.loc[d,s]))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc[:end]
S=pd.DataFrame(sig,columns=['date','symbol','signal']); S=S[S.date<=end]
mu=q.ic.mean(); sd=q.ic.std(ddof=1)
print('dates',len(q),'avg_instruments',round(q.n.mean(),3),'daily_IC',round(mu,6),'ICIR',round(mu/sd*np.sqrt(252),6),'hit',round((q.ic>0).mean(),4),'coverage',round(S.signal.notna().mean(),4),'active_condition',round(active.loc[:end].mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic
 print('regime',a+'-'+b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1)*np.sqrt(252),6) if len(z)>1 else None)
for h in [1,3,5]:
 yy=P.pct_change(h).shift(-h); rr=[]
 for d in P.index:
  x=f.loc[d]; z=yy.loc[d]; ok=x.notna()&z.notna()
  if ok.sum()>=8: rr.append(spearmanr(x[ok],z[ok]).statistic)
 print('decay',h,'d',len(rr),'IC',round(np.nanmean(rr),6))
S.to_csv('scripts/miner_3_20310821_breadth_dispersion_continuation_signal.csv',index=False)
