import pandas as pd, numpy as np, os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
start=pd.Timestamp('2026-07-16'); end=pd.Timestamp('2033-05-30'); D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[start:end]; r=px.pct_change()
# Compression-conditioned breakout: lagged 20d risk-adjusted trend, amplified when recent volatility is compressed.
vol20=r.rolling(20,min_periods=15).std(); vol60=r.rolling(60,min_periods=40).std()
compression=(vol20/vol60).clip(0.4,2.5)
sig=((px/px.shift(20)-1)/(vol20*np.sqrt(20))).div(compression).shift(1)
results={}
for h in [1,5,10,20,30]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(v): vals.append(v); ns.append(len(q))
 a=np.array(vals); results[h]=a
 print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f thirds=%s'%(h,len(a),np.mean(ns),np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12),np.mean(a>0),[round(np.mean(z),6) for z in np.array_split(a,3)]))
a=results[10]
print('coverage=%.4f turnover=%.4f n_assets=%d n_dates=%d'%(sig.notna().mean().mean(),(sig.rank(axis=1).diff().abs().stack()/15).mean(),len(D),len(px)))
sig.to_csv('scripts/miner_2_20330530_compression_breakout_signal.csv')
