import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=5000).set_index('date')['close'].astype(float) for s in syms}
P=pd.DataFrame(px).sort_index().ffill()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# Lag all information: percentile and price return are shifted before forming signal.
dxy_pct=dxy.rolling(252,min_periods=120).rank(pct=True).shift(1)
mom=P.pct_change(60).shift(1)
rel=mom.sub(mom.median(axis=1),axis=0)
# Strong dollar regime: favor defensive/commodity relative winners via contrarian relative momentum;
# otherwise follow 60d relative momentum.
F=rel.copy(); F.loc[dxy_pct>=.70,:]=-rel.loc[dxy_pct>=.70,:]
F=F.sub(F.median(axis=1),axis=0)
rows=[]
for h in [5,10,20,40]:
 fr=P.shift(-h)/P-1; ics=[]; ns=[]; ds=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): ics.append(c); ns.append(len(z)); ds.append(d)
 a=pd.Series(ics)
 if len(a):
  rows.append((h,len(a),np.mean(ns),np.mean(ns)/len(syms),a.mean(),a.mean()/a.std(),(a>0).mean(),min(ds).date(),max(ds).date()))
for r in rows: print('h=%d dates=%d avg_n=%.3f coverage=%.4f IC=%.8f ICIR=%.5f hit=%.4f start=%s end=%s'%r)
# regime split at selected 40d horizon for robustness
fr=P.shift(-40)/P-1
for label,mask in [('dxy_low',dxy_pct<.30),('dxy_mid',(dxy_pct>=.30)&(dxy_pct<.70)),('dxy_high',dxy_pct>=.70)]:
 a=[]
 for d in F.index[mask.fillna(False)]:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): a.append(c)
 a=pd.Series(a); print('regime=%s dates=%d IC=%.8f ICIR=%.5f'%(label,len(a),a.mean() if len(a) else np.nan,a.mean()/a.std() if len(a)>1 else np.nan))
F.to_csv('scripts/miner_2_20350927_dxy_conditioned_momentum_signal.csv',index_label='date')
