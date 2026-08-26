import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2035-07-08')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float).sort_index() for s in U}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).sort_index()
common=sorted(set.intersection(*[set(x.index) for x in px.values()]) & set(vix.index)); rows=[]; dates=[]; ns=[]; out=[]
for dt in common:
 lv=vix.index.get_loc(dt)
 if lv<125: continue
 vv=vix.iloc[lv]; v5=vv/vix.iloc[lv-5]-1; med=vix.iloc[lv-120:lv].median()
 vals={}; fw={}
 for s,p in px.items():
  loc=p.index.get_loc(dt)
  if loc<65 or loc+20>=len(p): continue
  r=p.pct_change(); vals[s]=(p.iloc[loc]/p.iloc[loc-20]-1,r.iloc[loc-60:loc].std()*np.sqrt(60)); fw[s]=p.iloc[loc+20]/p.iloc[loc]-1
 if len(vals)<8: continue
 medr=np.median([v[0] for v in vals.values()])
 # VIX impulse amplifier: positive recent volatility shock increases residual-reversal exposure, capped
 amp=1+min(max(v5,0),0.50) if vv>med else 1.0
 f={s:-(v[0]-medr)/max(v[1],1e-6)*amp for s,v in vals.items()}
 a=np.array(list(f.values())); b=np.array([fw[s] for s in f]); ic=spearmanr(a,b).statistic
 if np.isfinite(ic): rows.append(ic); dates.append(dt); ns.append(len(a))
 for s in vals: out.append({'date':dt.date().isoformat(),'symbol':s,'signal':f[s],'vix':vv,'vix5':v5,'gate':int(vv>med)})
x=np.array(rows); ds=np.array(dates,dtype='datetime64[ns]')
print('factor=VIX-impulse amplified residual reversal H20 cut',cut.date(),'dates',len(x),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.3f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-07-08'),('2034-07-01','2035-07-08')]:
 z=x[(ds>=np.datetime64(a))&(ds<=np.datetime64(b))]; print(a+' to '+b,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,round(np.mean(z>0),3))
pd.DataFrame(out).to_csv('scripts/miner_1_20350709_vix_impulse_reversal_signal.csv',index=False)
