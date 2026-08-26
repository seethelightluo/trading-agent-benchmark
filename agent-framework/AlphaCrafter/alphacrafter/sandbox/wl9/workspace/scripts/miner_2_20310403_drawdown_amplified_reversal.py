import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
root='../persistent/stock_data'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv(os.path.join(root,a+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 px[a]=d.close.replace(0,np.nan)
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change()
# Contrarian 20d risk-adjusted return, amplified by current 60d drawdown depth.
vol=ret.rolling(20,min_periods=16).std()*np.sqrt(252)
base=-ret.rolling(20,min_periods=16).sum()/vol
dd=prices/prices.rolling(60,min_periods=40).max()-1
factor=(base*(1+dd.abs())).shift(1)
for h in [5,10,20,40,60]:
 fwd=prices.shift(-h)/prices-1; ics=[]; ns=[]
 for dt in factor.index:
  ok=factor.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   v=spearmanr(factor.loc[dt,ok],fwd.loc[dt,ok]).statistic
   if np.isfinite(v): ics.append(v); ns.append(ok.sum())
 z=np.asarray(ics); print(h,'dates',len(z),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
# regime slices for admission horizon
h=20; fwd=prices.shift(-h)/prices-1; rows=[]
for dt in factor.index:
 ok=factor.loc[dt].notna()&fwd.loc[dt].notna()
 if ok.sum()>=8:
  v=spearmanr(factor.loc[dt,ok],fwd.loc[dt,ok]).statistic
  if np.isfinite(v): rows.append((dt,v,ok.sum()))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for name,sl in [('2020-22',r.loc['2020':'2022']),('2023-25',r.loc['2023':'2025']),('2026-28',r.loc['2026':'2028']),('2029-31',r.loc['2029':'2031'])]: print('regime',name,'n',len(sl),'ic',round(sl.ic.mean(),6))
out=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20310403_drawdown_amplified_reversal_signal.csv',index=False)
ranks=factor.rank(axis=1,pct=True); print('turnover_proxy',round(float(ranks.diff().abs().mean(axis=1).mean()),6),'signal_rows',len(out))
