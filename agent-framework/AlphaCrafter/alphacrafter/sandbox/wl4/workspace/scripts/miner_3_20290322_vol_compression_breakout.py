import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
acct=get_account_dict(); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in syms:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d)>150:
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); frames[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(frames).sort_index(); r=p.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
f=(p.pct_change(20)/(v20*np.sqrt(20))).mul(np.exp(-(v20/v60).clip(.2,3)-.5),axis=0).replace([np.inf,-np.inf],np.nan).clip(-8,8)
def calc(h):
 fr=f.shift(1); fw=p.pct_change(h).shift(-h); vals=[]; turns=[]; cov=[]; ns=[]
 for dt in fr.index:
  a=fr.loc[dt]; b=fw.loc[dt]; ok=a.notna()&b.notna(); n=ok.sum()
  if n>=8: vals.append(a[ok].corr(b[ok],method='spearman')); cov.append(ok.mean()); ns.append(n)
  j=fr.index.get_loc(dt)
  if j:
   prev=fr.iloc[j-1]; oo=a.dropna().index.intersection(prev.dropna().index)
   if len(oo)>=8: turns.append(1-a[oo].rank().corr(prev[oo].rank(),method='spearman'))
 z=pd.Series(vals).dropna(); ic=z.mean(); ir=ic/z.std(ddof=1)
 print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={ic:.5f} ICIR={ir:.5f} hit={(z>0).mean():.3f} coverage={np.mean(cov):.4f} rank_turn={np.nanmean(turns):.5f}')
 for n in [250,500]:
  q=z.tail(n); print(f' recent{n} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} dates={len(q)}')
for h in [1,5,10,20]: calc(h)
print('instruments',len(frames),'range',p.index.min().date(),p.index.max().date())
