import pandas as pd, numpy as np, glob, os, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
 f=os.path.join(base,a+'.csv'); d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); px[a]=d.set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Candidate: stress-conditioned short reversal, residualized cross-sectionally.
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); vix=vix.set_index('date').close.pct_change(5).reindex(p.index).ffill()
# Stress score uses only prior completed observations at date t; asset signal based through t.
csmed=r.median(axis=1); excess=r.sub(csmed,axis=0)
ret3=excess.rolling(3).sum(); vol20=r.rolling(20).std();
stress=(vix>vix.rolling(60).quantile(.65)).astype(float)
# continuous, interpretable: reversal amplified in elevated VIX, mild baseline otherwise
fac=-(ret3/vol20)*(.35+.65*stress.values[:,None])
# shift signal? forward return from next day; factor at t uses t data, valid.
rows=[]
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1
 ics=[]; ns=[]; turnovers=[]
 for dt in p.index:
  x=fac.loc[dt] if hasattr(fac,'loc') else fac[p.index.get_loc(dt)]
  y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 # turnover on 10d sampled ranks
 sig=pd.DataFrame(fac,index=p.index); ranks=sig.rank(axis=1,pct=True); turns=[]
 for i in range(10,len(ranks),10):
  a=ranks.iloc[i-10].dropna(); b=ranks.iloc[i].dropna(); common=a.index.intersection(b.index)
  if len(common)>=8: turns.append(np.mean(abs(a[common]-b[common])))
 ic=np.array(ics); rows.append((h,len(ic),np.nanmean(ic),np.nanstd(ic,ddof=1),np.nanmean(ic)/(np.nanstd(ic,ddof=1)+1e-12),np.mean(ic>0),np.mean(ns),np.mean(turns)))
print('candidate stress_amp_reversal; dates/instruments use >=8')
for x in rows: print('H%d dates=%d IC=%.6f ICIR=%.6f hit=%.3f meanN=%.2f turnover10=%.4f'% (x[0],x[1],x[2],x[4],x[5],x[6],x[7]))
for label,sub in [('all',p.index),('2020-23',p.index[(p.index.year<=2023)]),('2024-27',p.index[(p.index.year>=2024)&(p.index.year<=2027)]),('2028-30',p.index[(p.index.year>=2028)&(p.index.year<=2030)]),('2031+',p.index[p.index.year>=2031]),('recent120',p.index[-120:])]:
 h=1; fwd=p.shift(-h)/p-1; a=[]
 for dt in sub:
  z=pd.concat([pd.Series(fac[p.index.get_loc(dt)],index=p.columns),fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(label,'n=',len(a),'IC=%.6f ICIR=%.6f'%(np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12) if len(a)>1 else 0))
# library correlation audit common cell, compare factor expressions reconstructed approximately via factor JSON impossible; report against all factor stored signal metadata unavailable
print('coverage=%.4f'%(np.isfinite(fac).sum().sum()/fac.size))
