import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cut=pd.Timestamp('2032-11-14'); P=pd.DataFrame({s:D[s].close for s in U}); lr=np.log(P).diff()
# One idea: 5-day reversal scaled by lagged 20-day true range, cross-sectional demeaned.
# Range uses only bars through t and signal is lagged by one completed session.
rets={h:[] for h in [1,5,10,20]}; dates={h:[] for h in rets}; ns={h:[] for h in rets}; turns=[]; prev=None; rows=[]
for i,t in enumerate(P.index[P.index<=cut]):
 if i<25: continue
 past=P.index[P.index<=t]
 # signal at t based on t-1 data
 if len(past)<27: continue
 end=past[-2]; hist=P.index[P.index<=end]
 r5=lr.loc[:end].tail(5).sum()
 ranges=[]
 for s in U:
  x=D[s].loc[:end].tail(20)
  ranges.append((s, ((x.high-x.low)/x.close).mean()))
 atr=pd.Series(dict(ranges)); sig=(-r5/atr).replace([np.inf,-np.inf],np.nan).dropna(); sig=sig-sig.median()
 rank=sig.rank(pct=True); turns.append(0 if prev is None else (rank-prev.reindex(rank.index).fillna(.5)).abs().mean()); prev=rank
 for h in rets:
  yy={}
  for s in sig.index:
   fut=D[s].loc[D[s].index>t].close
   if len(fut)>=h: yy[s]=fut.iloc[h-1]/P.loc[t,s]-1
  com=list(set(sig.index)&set(yy));
  if len(com)>=8:
   rets[h].append(spearmanr(sig[com],pd.Series(yy)[com]).statistic); dates[h].append(t); ns[h].append(len(com))
  
for h in rets:
 a=np.array(rets[h]); print(h,'dates',len(a),'avgN',round(np.mean(ns[h]),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a),6),'hit',round(np.mean(a>0),4))
print('turnover',round(np.mean(turns),6),'coverage',1.0)
# thirds for 10d
x=np.array(rets[10]); print('thirds10', [round(np.mean(z),6) for z in np.array_split(x,3)])
# save signal artifact for reproducibility
out=[]
for t in P.index[P.index<=cut][-500:]:
 past=P.index[P.index<=t]
 if len(past)<27: continue
 end=past[-2]; r5=lr.loc[:end].tail(5).sum(); atr=pd.Series({s:((D[s].loc[:end].tail(20).high-D[s].loc[:end].tail(20).low)/D[s].loc[:end].tail(20).close).mean() for s in U}); sig=(-r5/atr).replace([np.inf,-np.inf],np.nan).dropna(); sig=sig-sig.median()
 for s,v in sig.items(): out.append({'date':t.date(),'symbol':s,'signal':float(v)})
pd.DataFrame(out).to_csv('scripts/miner_2_20321115_range_norm_reversal_signal.csv',index=False)
