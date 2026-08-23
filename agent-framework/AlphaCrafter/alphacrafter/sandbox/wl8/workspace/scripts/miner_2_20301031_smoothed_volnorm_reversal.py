import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
    try:
        d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
        raw[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
    except FileNotFoundError: pass
px=pd.DataFrame(raw).sort_index().loc[:'2030-10-30']
ret=px.pct_change(); vol=ret.rolling(20,min_periods=15).std()
base=-(px/px.shift(5)-1)/vol
# Three-day trailing mean of already observable normalized signals, reducing noise/turnover
sig=base.rolling(3,min_periods=2).mean()
fwd=px.shift(-10)/px-1
rows=[]
for dt in sig.index:
 z,y=sig.loc[dt],fwd.loc[dt]; ok=z.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,z[ok].corr(y[ok],method='spearman'),int(ok.sum())))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stat(q):
 q=q.dropna(); return len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
print('candidate=smoothed_volnorm_reversal; dates',len(r),'period',r.index.min(),r.index.max(),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
for lab,q in [('all',r.ic),('recent180',r.ic.tail(180)),('recent360',r.ic.tail(360)),('2029',r.loc['2029':'2029'].ic),('2030',r.loc['2030':'2030'].ic)]: print(lab,stat(q))
rank=sig.rank(axis=1,pct=True); print('turnover',((rank-rank.shift(1)).abs().mean(axis=1)).dropna().mean())
for h in [5,10,20]:
 yy=px.shift(-h)/px-1;a=[]
 for dt in sig.index:
  z,y=sig.loc[dt],yy.loc[dt];ok=z.notna()&y.notna()
  if ok.sum()>=8:a.append(z[ok].corr(y[ok],method='spearman'))
 print('horizon',h,'n',len(a),'ic',np.nanmean(a),'icir',np.nanmean(a)/np.nanstd(a,ddof=1))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20301031_smoothed_volnorm_reversal_signal.csv',index=False)
r.reset_index().to_csv('scripts/miner_2_20301031_smoothed_volnorm_reversal_ic.csv',index=False)
