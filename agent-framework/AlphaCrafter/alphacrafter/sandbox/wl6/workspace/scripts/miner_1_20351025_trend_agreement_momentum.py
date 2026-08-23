import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,6000)
 if d is None or len(d)==0: d=get_index_daily_data(s,6000)
 return None if d is None or len(d)==0 else d[['date','close']].drop_duplicates('date').set_index('date')['close'].astype(float)
p={s:load(s) for s in U}; p={s:x for s,x in p.items() if x is not None}
P=pd.DataFrame(p).sort_index().ffill().loc[:pd.Timestamp('2035-10-24')]
r=P.pct_change(); mom=P/P.shift(20)-1; vol=r.rolling(60,min_periods=40).std()
# Trend-agreement quality: 20d momentum scaled by risk, retained only when 5/20/60d directions agree.
m5=P/P.shift(5)-1; m60=P/P.shift(60)-1
agree=((np.sign(m5)==np.sign(mom)) & (np.sign(mom)==np.sign(m60))).astype(float)
sig=(mom/(vol+1e-8))*agree.replace(0,np.nan).shift(1)
fw=P.shift(-10)/P-1
vals=[]; ns=[]; dates=[]
for d in sig.index:
 ok=sig.loc[d].notna()&fw.loc[d].notna()
 if ok.sum()>=8:
  q=sig.loc[d,ok].corr(fw.loc[d,ok],method='spearman')
  if pd.notna(q): vals.append(q); ns.append(ok.sum()); dates.append(d)
a=pd.Series(vals,index=dates)
print('h=10 dates=%d avg_inst=%.3f IC=%.8f ICIR=%.8f hit=%.4f'%(len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(len(a)),(a>0).mean()))
for h in [5,20,40]:
 fw=P.shift(-h)/P-1; v=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   q=sig.loc[d,ok].corr(fw.loc[d,ok],method='spearman')
   if pd.notna(q): v.append(q)
 z=pd.Series(v); print('h=%d dates=%d IC=%.8f ICIR=%.8f'%(h,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z))))
rank=sig.rank(axis=1,pct=True); print('coverage=%.6f turnover=%.6f active_dates=%d instruments=%d rows=%d end=%s'%(sig.notna().sum().sum()/(len(sig)*len(U)),rank.diff().abs().mean().mean(),sig.notna().any(axis=1).sum(),len(U),len(sig),P.index.max().date()))
# artifact for deterministic screening
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20351025_trend_agreement_momentum_signal.csv',index=False)
