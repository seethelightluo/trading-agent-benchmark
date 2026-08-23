import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index(); r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); F=(-(P/P.shift(10)-1)/(vol*np.sqrt(252))).shift(1); R=P.shift(-10)/P-1
rows=[]; prev=None
for d in F.index:
 x,y=F.loc[d],R.loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  from scipy.stats import spearmanr
  ic=spearmanr(x[ok],y[ok]).statistic; rank=x.rank(pct=True); rows.append((d,ic,ok.sum(),ok.mean(),np.mean(np.abs(rank-(prev if prev is not None else rank))))); prev=rank
q=pd.DataFrame(rows,columns=['date','ic','n','coverage','turnover']).set_index('date'); print('factor=lagged negative 10d return / 20d annualized vol; horizon=10d');print('dates',len(q),'avgN',q.n.mean(),'coverage',q.coverage.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/(q.ic.std(ddof=1)/np.sqrt(len(q))),'hit', (q.ic>0).mean(),'turnover',q.turnover.mean())
for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027YTD','2027-01-01','2027-06-02')]:
 z=q.loc[a:b].ic; print(name,'dates',len(z),'IC',z.mean() if len(z) else np.nan,'ICIR',z.mean()/(z.std(ddof=1)/np.sqrt(len(z))) if len(z)>1 else np.nan)
F.reset_index().to_csv('scripts/miner_2_20270603_reversal10d_signal.csv',index=False)
