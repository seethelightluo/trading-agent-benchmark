import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-12-01'); P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date)
 P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Trend continuation is emphasized after lagged volatility compression; all inputs are lagged.
trend=px.pct_change(5).shift(1); vol=r.rolling(20).std().shift(1); base=vol.rolling(60).median().shift(1)
sig=trend/(vol/base).clip(.25,2.0)
fwd=px.shift(-1)/px-1
rows=[]
for d in sig.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  z=spearmanr(g.s,g.f).statistic
  if np.isfinite(z): rows.append((d,z,len(g)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stat(q):
 z=a.loc[q,'ic']; return len(z),round(a.loc[q,'n'].mean(),2),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round((z>0).mean(),4)
print('data_end',px.index.max().date(),'dates',len(a),'rows',int(a.n.sum()),'overall',stat(slice(None)))
y=a.index.year
for name,q in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',a.index>=END-pd.Timedelta(days=180))]: print(name,stat(q))
print('coverage',round(sig.notna().sum().sum()/sig.size,4),'turnover_proxy',round((sig.rank(axis=1,pct=True).diff().abs().mean().mean()),4))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20271202_compressed_trend_signal.csv',index=False)
