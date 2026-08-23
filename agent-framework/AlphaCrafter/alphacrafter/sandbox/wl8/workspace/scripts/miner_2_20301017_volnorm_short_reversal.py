import numpy as np, pandas as pd

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: short-horizon reversal normalized by medium volatility. Signal at t uses only closes <=t.
raw={}
for s in U:
    try:
        d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
        x=d.drop_duplicates('date').set_index('date').close.astype(float)
        raw[s]=x
    except FileNotFoundError: pass
px=pd.DataFrame(raw).sort_index(); cutoff=pd.Timestamp('2030-10-17'); px=px.loc[px.index<=cutoff]
ret=px.pct_change()
# 5d reversal / 20d annualized-ish vol; winsorization is cross-sectional only
sig=-(px/px.shift(5)-1) / ret.rolling(20,min_periods=15).std()
# forward 10d return, strictly after t
fwd=px.shift(-10)/px-1
rows=[]
for dt in sig.index:
    z=sig.loc[dt]; y=fwd.loc[dt]; ok=z.notna()&y.notna()
    if ok.sum()>=8:
        rows.append((dt,z[ok].corr(y[ok],method='spearman'),int(ok.sum())))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stat(q):
    q=q.dropna(); return (len(q),q.mean(),q.mean()/q.std(ddof=1), (q>0).mean()) if len(q)>1 else (len(q),np.nan,np.nan,np.nan)
# turnover measured rank changes on common universe
rank=sig.rank(axis=1,pct=True)
to=(rank-rank.shift(1)).abs().mean(axis=1).dropna().mean()
print('candidate=volnorm_short_reversal; dates',len(r),'period',r.index.min(),r.index.max(),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
for label,q in [('all',r.ic),('recent180',r.ic.tail(180)),('recent360',r.ic.tail(360)),('2029',r.loc['2029':'2029'].ic),('2030',r.loc['2030':'2030'].ic),('5d',[])]:
    if label=='5d': continue
    print(label,stat(q))
print('turnover',to)
# decay horizons using same formation signal
for h in [5,10,20]:
    yy=px.shift(-h)/px-1; a=[]
    for dt in sig.index:
        z=sig.loc[dt]; y=yy.loc[dt]; ok=z.notna()&y.notna()
        if ok.sum()>=8:a.append(z[ok].corr(y[ok],method='spearman'))
    print('horizon',h,'n',len(a),'ic',np.nanmean(a),'icir',np.nanmean(a)/np.nanstd(a,ddof=1))
# artifact for downstream audit
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20301017_volnorm_short_reversal_signal.csv',index=False)
r.reset_index().to_csv('scripts/miner_2_20301017_volnorm_short_reversal_ic.csv',index=False)
