import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2032-12-22')
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).loc[:end]
r=px.pct_change(); b=r.mean(axis=1); res=r.sub(b,axis=0)
disp=res.std(axis=1); gate=disp>disp.rolling(120,min_periods=80).median()
# short-horizon residual shock reversal, normalized by recent total vol
f=(-res.rolling(5,min_periods=5).sum()/ (res.rolling(30,min_periods=20).std()+1e-8)).where(gate,0).shift(1)
y=px.shift(-10).div(px)-1; out=[]
for d in f.index:
 a=f.loc[d]; z=y.loc[d]; ok=a.notna()&z.notna()&np.isfinite(a)&np.isfinite(z)
 if ok.sum()>=8:
  ic=a[ok].rank().corr(z[ok],method='spearman')
  if np.isfinite(ic): out.append((d,ic,ok.sum()))
r=pd.DataFrame(out,columns=['date','ic','n']).set_index('date').loc[:end-pd.Timedelta(days=15)]
print('dates',len(r),'avg_n',r.n.mean(),'coverage',f.notna().sum(axis=1).mean()/15,'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean(),'turnover',f.rank(axis=1).diff().abs().mean().mean()/14)
for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-12-10')]:
 q=r.loc[a:b].ic; print(a[:4]+'-'+b[:4],len(q),q.mean(),q.mean()/q.std(ddof=1))
out='scripts/miner_2_20321223_shock_reversal_signal.csv'; f.to_csv(out); print('artifact',out)
