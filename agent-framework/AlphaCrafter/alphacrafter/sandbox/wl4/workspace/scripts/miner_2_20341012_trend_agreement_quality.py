import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in symbols:
    d=get_stock_daily_data(s,days=5200)
    if d is not None and len(d)>100: raw[s]=d.copy().set_index('date')['close'].astype(float)
px=pd.DataFrame(raw).sort_index(); rets=px.pct_change()
r5=px.pct_change(5); r20=px.pct_change(20); r60=px.pct_change(60)
v20=rets.rolling(20).std()*np.sqrt(20)
z5=r5/v20; z20=r20/(rets.rolling(20).std()*np.sqrt(20)); z60=r60/(rets.rolling(60).std()*np.sqrt(60))
fac=((z5+z20+z60)/3 - .35*(z5-z20).abs() - .35*(z20-z60).abs()).shift(1)
fwd=px.shift(-10)/px-1
rows=[]; sigrows=[]
for dt in fac.index:
 x=fac.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append([dt,x[ok].corr(y[ok],method='spearman'),x[ok].corr(y[ok]),ok.sum()])
 for s in fac.columns:
  if pd.notna(fac.loc[dt,s]): sigrows.append([dt,s,fac.loc[dt,s]])
ic=pd.DataFrame(rows,columns=['date','rank_ic','ic','n']).set_index('date')
for w in [None,120,260,520,780]:
 q=ic if w is None else ic.tail(w); mean=q.rank_ic.mean(); sd=q.rank_ic.std(ddof=1)
 print('window',w or 'all','dates',len(q),'avg_n',q.n.mean(),'IC',mean,'ICIR',mean/sd*np.sqrt(252/10) if sd else np.nan,'hit',(q.rank_ic>0).mean())
ss=pd.DataFrame(sigrows,columns=['date','symbol','value']).pivot(index='date',columns='symbol',values='value'); ranks=ss.rank(axis=1,pct=True)
print('coverage',len(sigrows)/(len(fac.index)*len(symbols)),'turnover',(ranks.diff().abs().mean(axis=1)/2).mean(),'symbols',len(raw),'dates',len(px))
ic.to_csv('scripts/artifacts/miner_2_20341012_trend_agreement_ic.csv'); ss.to_csv('scripts/artifacts/miner_2_20341012_trend_agreement_signal.csv')
