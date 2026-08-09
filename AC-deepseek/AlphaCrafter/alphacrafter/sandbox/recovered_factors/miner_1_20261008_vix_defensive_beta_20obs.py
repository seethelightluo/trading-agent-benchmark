"""One candidate: VIX defensive beta (20 observations), validated to prior completed day."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-10-07')
def load(s):
    for root in ('../persistent/stock_data','../persistent/index_data'):
        p=f'{root}/{s}.csv'
        if os.path.exists(p):
            d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
            return d.loc[:END]
    raise FileNotFoundError(s)
raw={s:load(s) for s in A}; close=pd.concat({s:d.close for s,d in raw.items()},axis=1,sort=True); ret=close.pct_change()
vix=load('VIX').close.pct_change()
# Higher score means an asset's last-20-observation returns were less positively linked to VIX shocks.
factor=pd.concat({s:-ret[s].rolling(20,min_periods=15).corr(vix) for s in A},axis=1)
def metrics(h, lo='2020-01-01', hi='2026-10-07'):
    f=factor.loc[lo:hi]; fw=(close.shift(-h)/close-1).reindex(f.index); ic=[]; n=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
            if np.isfinite(x): ic.append(x); n.append(len(z))
    x=np.array(ic); return len(x),x.mean() if len(x) else np.nan,(x.mean()/x.std(ddof=1) if len(x)>1 else np.nan),(x>0).mean() if len(x) else np.nan,np.mean(n) if n else np.nan
print('CANDIDATE vix_defensive_beta_20obs; endpoint',END.date())
print('coverage',factor.notna().stack().mean(),'valid_cells',factor.notna().sum().sum(),'total_cells',factor.size)
for h in (1,5,10,20): print('H',h,'dates IC ICIR hit avg_instruments',metrics(h))
for name,lo,hi in [('pre-VIX-history','2020-01-01','2024-12-31'),('recent','2025-01-01','2026-10-07')]: print('REGIME H10',name,metrics(10,lo,hi))
r=factor.rank(axis=1,pct=True); print('rank_turnover',r.diff().abs().stack().mean())
# Reconstruct all currently admitted signals, then test pooled asset-date Spearman overlap.
lib={}
for path in glob.glob('factors/*.json'):
    if path.endswith('.bak'): continue
    fid=json.load(open(path))['factor_id']
    if 'relative_volume' in fid: L=pd.concat({s:np.log(raw[s].volume/raw[s].volume.rolling(20,min_periods=15).mean()) for s in A},axis=1)
    elif 'volscaled_reversal_1obs' in fid: L=-ret/ret.rolling(20,min_periods=15).std()
    elif 'volnorm_reversal' in fid: L=-(close.pct_change(5))/ret.rolling(5,min_periods=4).std()
    else: L=(close.pct_change(20))/ret.rolling(20,min_periods=15).std()
    z=pd.concat([factor.stack().rename('candidate'),L.stack().rename('library')],axis=1).dropna()
    rho=z.candidate.corr(z.library,method='spearman'); lib[fid]=(rho,len(z)); print('LIBCORR',fid,'rho',rho,'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',max(abs(x[0]) for x in lib.values()))
