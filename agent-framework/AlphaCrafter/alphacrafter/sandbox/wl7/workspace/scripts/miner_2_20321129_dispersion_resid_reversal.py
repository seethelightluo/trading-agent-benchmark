import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cut=pd.Timestamp('2032-11-28'); P=pd.DataFrame({s:D[s].close for s in U}); lr=np.log(P).diff();
# Candidate: dispersion-gated 5d residual reversal, scaled by idiosyncratic vol.
# Gate is high cross-sectional absolute 5d return dispersion, all inputs lagged through t-1.
IC={h:[] for h in [1,5,10,20]}; Ns={h:[] for h in IC}; prev=None; turns=[]; rows=[]
for t in P.index[P.index<=cut]:
 i=P.index.get_loc(t)
 if i<65: continue
 end=P.index[i-1]; r5=lr.loc[:end].tail(5).sum(); vol=lr.loc[:end].tail(30).std(); disp=r5.dropna().abs().std()
 # high dispersion relative to trailing 60-session dispersion distribution
 dseries=lr.abs().sum(axis=1).rolling(5).std().loc[:end].dropna()
 if len(dseries)<40 or disp <= dseries.tail(40).quantile(.60): continue
 resid=r5-r5.median(); sig=(-resid/vol).replace([np.inf,-np.inf],np.nan).dropna(); sig=sig-sig.median()
 rank=sig.rank(pct=True); turns.append(0 if prev is None else (rank-prev.reindex(rank.index).fillna(.5)).abs().mean()); prev=rank
 for h in IC:
  fut={s:D[s].loc[D[s].index>t].close.iloc[h-1]/P.loc[t,s]-1 for s in sig.index if len(D[s].loc[D[s].index>t])>=h}
  c=list(set(sig.index)&set(fut))
  if len(c)>=8: IC[h].append(spearmanr(sig[c],pd.Series(fut)[c]).statistic); Ns[h].append(len(c))
 for s,v in sig.items(): rows.append({'date':t.date(),'symbol':s,'signal':float(v)})
for h,a0 in IC.items():
 a=np.array(a0); print('H',h,'dates',len(a),'avgN',round(np.mean(Ns[h]),2) if a.size else 0,'IC',round(np.nanmean(a),6) if a.size else None,'ICIR',round(np.nanmean(a)/np.nanstd(a),6) if a.size else None,'hit',round(np.mean(a>0),4) if a.size else None)
 if h==10 and a.size: print('thirds10',[round(np.mean(x),6) for x in np.array_split(a,3)])
print('turnover',round(np.mean(turns),6) if turns else None,'coverage',round(np.mean([len(x) for x in []]),3) if False else 'conditional')
pd.DataFrame(rows).to_csv('scripts/miner_2_20321129_dispersion_resid_reversal_signal.csv',index=False)
