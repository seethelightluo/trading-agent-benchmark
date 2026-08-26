import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cut=pd.Timestamp('2032-05-30'); P=pd.DataFrame({s:D[s].close for s in U}); lr=np.log(P).diff()
# Candidate: downside-risk-adjusted 20d trend, with cross-sectional demeaning.
horizons=[1,5,10,20]; R={h:[] for h in horizons}; ns={h:[] for h in horizons}; cov=[]; turn=[]; prev=None; rows=[]
for t in P.index[P.index<=cut]:
    r20=lr.loc[:t].tail(20).sum(); x=lr.loc[:t].tail(40)
    down=x.where(x<0,0).std(); sig=(r20/down).replace([np.inf,-np.inf],np.nan).dropna()
    sig=sig[sig.index.isin(U)]
    if len(sig)<8: continue
    sig=sig-sig.median(); cov.append(len(sig)/15); rank=sig.rank(pct=True)
    turn.append(0 if prev is None else (rank-prev.reindex(rank.index).fillna(.5)).abs().mean()); prev=rank
    for s,v in sig.items(): rows.append({'date':t.date(),'symbol':s,'signal':v})
    for h in horizons:
      yy={}
      for s in sig.index:
        fut=D[s].loc[D[s].index>t].close
        if len(fut)>=h: yy[s]=fut.iloc[h-1]/D[s].loc[:t].close.iloc[-1]-1
      com=list(set(sig.index)&set(yy))
      if len(com)>=8:
        q=spearmanr(sig[com],pd.Series(yy)[com]).statistic
        if np.isfinite(q): R[h].append(q); ns[h].append(len(com))
print('cutoff',cut.date(),'universe',len(U),'valid_dates',len(cov),'coverage',round(np.mean(cov),5),'turnover',round(np.mean(turn),5))
for h in horizons:
 q=pd.Series(R[h]); print('H',h,'dates',len(q),'avgN',round(np.mean(ns[h]),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
pd.DataFrame(rows).to_csv('scripts/miner_2_20320531_downside_trend_signal.csv',index=False)
