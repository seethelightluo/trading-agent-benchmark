import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; end='2031-08-21'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); R=P.pct_change(); r20=P.pct_change(20); resid=r20.sub(r20.median(axis=1),axis=0); y=P.pct_change().shift(-1)
# Mild, interpretable breadth/dispersion continuation: signal only when breadth is above its expanding median.
breadth=(R>0).sum(axis=1)/len(U); disp=R.std(axis=1); active=(breadth.rolling(5).mean()>breadth.rolling(252,min_periods=126).median())
f=resid.where(active,np.nan); rows=[]; sig=[]
for d in P.index:
 x=f.loc[d]; z=y.loc[d]; ok=x.notna()&z.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],z[ok]).statistic
  if np.isfinite(ic): rows.append((d,ic,ok.sum()))
 for s in U:sig.append((d,s,f.loc[d,s]))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc[:end]; S=pd.DataFrame(sig,columns=['date','symbol','signal']);S=S[S.date<=end];mu=q.ic.mean();sd=q.ic.std(ddof=1)
print('dates',len(q),'avgN',q.n.mean(),'IC',mu,'ICIR',mu/sd*np.sqrt(252),'hit',(q.ic>0).mean(),'coverage',S.signal.notna().mean(),'active',active.loc[:end].mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic;print(a+'-'+b,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252))
for h in [3,5]:
 yy=P.pct_change(h).shift(-h); rr=[]
 for d in P.index:
  x=f.loc[d];z=yy.loc[d];ok=x.notna()&z.notna()
  if ok.sum()>=8:rr.append(spearmanr(x[ok],z[ok]).statistic)
 print('decay',h,len(rr),np.nanmean(rr))
S.to_csv('scripts/miner_3_20310821_breadth_conditioned_momentum_signal.csv',index=False)
