import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2034-02-15')
xs={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 d=d.loc[:end]
 # overnight return: today's open / prior close - 1; intraday return: close/open - 1
 overnight=d['open']/d['close'].shift(1)-1
 intraday=d['close']/d['open']-1
 # divergence: recent intraday strength relative to overnight strength, volatility scaled
 sig=(intraday.rolling(5).mean()-overnight.rolling(5).mean())
 vol=d['pct_change'].rolling(20).std()*np.sqrt(252)
 xs[s]=pd.DataFrame({'f':sig/vol.replace(0,np.nan),'p':d['close']})
# aligned dates and forward 10 trading observations within each asset
rows=[]
for s,z in xs.items():
 z=z.dropna(); z['fwd']=z.p.shift(-10)/z.p-1; z=z.dropna()
 for dt,r in z.iterrows(): rows.append((dt,s,r.f,r.fwd))
a=pd.DataFrame(rows,columns=['date','symbol','f','fwd'])
ics=[]; ns=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fwd.nunique()>1:
  ics.append(spearmanr(g.f,g.fwd).statistic); ns.append(len(g))
ics=np.array(ics)
print('dates',len(ics),'avg_n',np.mean(ns),'assets',len(U),'coverage',len(a)/(len(ics)*len(U)))
for h in [5,10,20,40]:
 rr=[]
 for s,z in xs.items():
  z=z.dropna(); z['fwd']=z.p.shift(-h)/z.p-1; z=z.dropna()
  for dt,r in z.iterrows(): rr.append((dt,s,r.f,r.fwd))
 q=pd.DataFrame(rr,columns=['date','symbol','f','r']); out=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: out.append(spearmanr(g.f,g.r).statistic)
 out=np.array(out); print('horizon',h,'IC',np.nanmean(out),'ICIR',np.nanmean(out)/np.nanstd(out,ddof=1)*np.sqrt(252),'hit',np.mean(out>0))
# turnover proxy daily rank changes
ranks=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank')
print('turnover_proxy',ranks.diff().abs().mean().mean())
print('period',a.date.min(),a.date.max())
