import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; idx='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); R=P.pct_change(); vix=pd.read_csv(f'{idx}/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill(); r5=P.pct_change(5); med=r5.median(axis=1); resid=r5.sub(med,axis=0); breadth=(R>0).sum(axis=1)/15; active=(vix>vix.rolling(252,min_periods=126).quantile(.70))&(breadth.rolling(5).mean()<.45); f=-resid; f[~active]=np.nan; y=P.pct_change().shift(-1)
rows=[]; signals=[]
for d in P.index:
 x=f.loc[d]; z=y.loc[d]; ok=x.notna()&z.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],z[ok]).statistic
  if np.isfinite(ic): rows.append((d,ic,ok.sum()))
 for s in U: signals.append((d,s,f.loc[d,s]))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc[:'2031-08-20']; S=pd.DataFrame(signals,columns=['date','symbol','signal']); S=S[S.date<='2031-08-20']; mu=q.ic.mean(); sd=q.ic.std(ddof=1); print('active dates',len(q),'avg instruments',q.n.mean(),'IC',mu,'ICIR',mu/sd*np.sqrt(252),'hit',(q.ic>0).mean(),'coverage',S.signal.notna().mean());
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic; print(a+'-'+b,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252))
S.to_csv('scripts/miner_1_20310821_macro_breadth_shock_signal.csv',index=False)
