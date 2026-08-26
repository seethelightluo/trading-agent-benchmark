import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}); lr=np.log(P).diff(); cut=pd.Timestamp('2032-12-12')
# Candidate: downside-risk-scaled medium-term reversal, lagged by one session.
IC={h:[] for h in [1,5,10,20]}; Ns={h:[] for h in IC}; prev=None; turns=[]; rows=[]
for t in P.index[P.index<=cut]:
 i=P.index.get_loc(t)
 if i<55: continue
 end=P.index[i-1]; r10=lr.loc[:end].tail(10).sum(); x=lr.loc[:end].tail(40)
 down=x.where(x<0).std(); sig=(-r10/down).replace([np.inf,-np.inf],np.nan).dropna(); sig=sig-sig.median()
 rank=sig.rank(pct=True); turns.append(0 if prev is None else (rank-prev.reindex(rank.index).fillna(.5)).abs().mean()); prev=rank
 for h in IC:
  fut={s:D[s].loc[D[s].index>t].close.iloc[h-1]/P.loc[t,s]-1 for s in sig.index if len(D[s].loc[D[s].index>t])>=h}
  c=list(set(sig.index)&set(fut))
  if len(c)>=8: IC[h].append(spearmanr(sig[c],pd.Series(fut)[c]).statistic); Ns[h].append(len(c))
 for s,v in sig.items(): rows.append({'date':t.date(),'symbol':s,'signal':float(v)})
for h,a0 in IC.items():
 a=np.array(a0,dtype=float); print('H',h,'dates',len(a),'avgN',round(np.mean(Ns[h]),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a),6),'hit',round(np.mean(a>0),4))
 if h in [10,20] and len(a): print('thirds',[round(float(np.nanmean(x)),6) for x in np.array_split(a,3)])
print('turnover',round(np.mean(turns),6),'coverage',round(len(rows)/(len(P.index[P.index<=cut])-55)/15,4),'artifact','scripts/miner_2_20321213_downside_adj_reversal_signal.csv')
pd.DataFrame(rows).to_csv('scripts/miner_2_20321213_downside_adj_reversal_signal.csv',index=False)
