"""miner_3 one-idea validation: rolling downside-tail asymmetry (20d return skewness)."""
import glob
import numpy as np
import pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=(1,5,10,20)
F={};FW={};LIB={}
for a in ASSETS:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index();p=pd.to_numeric(d.close,errors='coerce');r=p.pct_change(fill_method=None)
 # Positive skew means upside-tail dominated recent return distribution; it may signal resilient demand rather than merely level momentum.
 F[a]=r.rolling(20,min_periods=15).skew();FW[a]={h:p.shift(-h)/p-1 for h in H}
 LIB.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
 LIB.setdefault('miner_3_relative_volume_participation_20d',{})[a]=np.log(pd.to_numeric(d.volume,errors='coerce')/pd.to_numeric(d.volume,errors='coerce').rolling(20,min_periods=15).mean())
 LIB.setdefault('miner_1_ravmom_20obs',{})[a]=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std();LIB.setdefault('miner_2_realized_volatility_20obs',{})[a]=r.rolling(20,min_periods=15).std();LIB.setdefault('miner_1_volnorm_reversal_5obs',{})[a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
f=pd.DataFrame(F).sort_index();print('FACTOR: return_skewness_20d = rolling skewness of 20 daily returns; high denotes upside-tail asymmetric recent returns');print('history',f.index.min().date(),f.index.max().date(),'instruments',len(ASSETS))
def st(x):return x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),x.std(ddof=1)/np.sqrt(len(x))
def calc(h):
 fw=pd.DataFrame({a:FW[a][h] for a in ASSETS});o=[];c=[]
 for dt in f.index:
  z=pd.DataFrame({'s':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(z)>=8:o.append((dt,z.s.corr(z.r,method='spearman')));c.append(len(z)/15)
 return pd.Series(dict(o)),np.mean(c)
for h in H:
 x,c=calc(h);m,ir,hit,se=st(x);print(f'h={h} dates={len(x)} meanIC={m:.6f} ICIR={ir:.6f} hit={hit:.4f} IC_se={se:.6f} coverage={c:.4f}')
 for nm,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  q=x[mask];a,b,c1,_=st(q);print(f'  {nm}: n={len(q)} IC={a:.6f} ICIR={b:.6f} hit={c1:.4f}')
r=f.rank(axis=1,pct=True);t=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover_rank_change={np.mean(t):.6f}; valid_signal_cells={f.notna().sum().sum()}/{f.size} ({f.notna().mean().mean():.4f})')
mx=0
for n,v in LIB.items():
 z=pd.concat([f.stack().rename('candidate'),pd.DataFrame(v).stack().rename('library')],axis=1).dropna();rho=z.candidate.corr(z.library,method='spearman');mx=max(mx,abs(rho));print(f'library {n}: rho={rho:.6f} cells={len(z)}')
print(f'LIBRARY records={len(glob.glob("factors/*.json"))} max_abs_library_correlation={mx:.6f}')
