"""miner_3: validate cross-asset peer-correlation dispersion, 20 completed observations.
High score means an asset has unusually heterogeneous relationships across the other 14 assets.
Uses only rows through the prior completed date (2026-09-23)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-09-23'); px={}; vv={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 px[a]=pd.to_numeric(d.close,errors='coerce').replace(0,np.nan); vv[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(fill_method=None); v=pd.DataFrame(vv).reindex(p.index).ffill()
vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/vol; rav=trend.copy(); rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); relvol=np.log(v/v.rolling(20,min_periods=15).mean())
peer=pd.DataFrame(index=p.index,columns=A,dtype=float); disp=peer.copy()
for a in A:
 c=pd.concat([r[a].rolling(20,min_periods=15).corr(r[b]) for b in A if b!=a],axis=1)
 peer[a]=c.mean(axis=1); disp[a]=c.std(axis=1,ddof=0)
lib={'risk_adjusted_trend_20d':trend,'ravmom_20obs':rav,'volnorm_reversal_5obs':rev,'realized_volatility_20obs':vol,'peer_crowding_correlation_20obs':peer,'relative_volume_participation_20d':relvol}
def ic(h):
 fw=p.shift(-h)/p-1; z=[]; ns=[]
 for dt in disp.index:
  ok=disp.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8: z.append((dt,spearmanr(disp.loc[dt,ok],fw.loc[dt,ok]).statistic));ns.append(ok.sum())
 return pd.Series(dict(z)),np.mean(ns)
print('FACTOR peer_correlation_dispersion_20d: std across 14 pairwise 20d return correlations; high=heterogeneous peer linkage')
print('validation_end',END.date(),'history',p.index.min().date(),p.index.max().date(),'assets',len(A),'signal_cells',disp.notna().sum().sum(),'/',disp.size,'coverage',round(disp.notna().mean().mean(),6))
ans={}
for h in [1,5,10,20]:
 s,n=ic(h);ans[h]=s; print('h',h,'dates',len(s),'meanIC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'IC_se',round(s.std(ddof=1)/np.sqrt(len(s)),6),'mean_n',round(n,2))
for label,mask in [('2020',ans[10].index.year==2020),('2021_22',ans[10].index.year.isin([2021,2022])),('2023_24',ans[10].index.year.isin([2023,2024])),('2025_26',ans[10].index.year.isin([2025,2026]))]:
 s=ans[10][mask];print('regime',label,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
rk=disp.rank(axis=1,pct=True); print('turnover_mean_abs_rank_change',round(rk.diff().abs().mean(axis=1).mean(),6))
mx=0
for n,x in lib.items():
 z=pd.concat([disp.stack(),x.stack()],axis=1).dropna();rho=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic;mx=max(mx,abs(rho));print('library',n,'rho',round(rho,6),'cells',len(z))
print('max_abs_library_correlation',round(mx,6))
