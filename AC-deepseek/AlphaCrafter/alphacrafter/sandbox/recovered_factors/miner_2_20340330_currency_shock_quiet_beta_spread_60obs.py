"""Miner_2: validate a currency-shock asymmetry exposure factor using completed bars only."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=list(get_account_dict()['watch_list'])
END='2034-03-29'
def ret(x):
    c=pd.to_numeric(x['close'],errors='coerce')
    return c.pct_change().replace([np.inf,-np.inf],np.nan)
def fetch(a,macro=False):
    try: return ret((get_index_daily_data if macro else get_stock_daily_data)(a,days=3800))
    except Exception as e: print('FETCH_FAIL',a,type(e).__name__); return pd.Series(dtype=float)
# A signed, direction-normalized USD basket: USDJPY and USDCNY rise = USD stronger; EURUSD is inverted.
fx=pd.concat([fetch('USDJPY',True),fetch('USDCNY',True),-fetch('EURUSD',True)],axis=1).median(axis=1)
r=pd.DataFrame({a:fetch(a) for a in A})
ix=r.index.intersection(fx.index); r=r.loc[ix]; fx=fx.loc[ix]
def beta(y,x):
    z=pd.concat([y,x],axis=1).dropna()
    if len(z)<8:return np.nan
    v=z.iloc[:,1].var()
    return z.iloc[:,0].cov(z.iloc[:,1])/v if v>1e-14 else np.nan
# At t, recent currency-shock beta minus quiet-currency beta, measured strictly through t.
sig=pd.DataFrame(np.nan,index=r.index,columns=A)
for pos in range(60,len(r)):
    sl=slice(pos-60,pos) # completed history ending t-1
    f=fx.iloc[sl]; shock=f.abs()>f.abs().median()
    for a in A:
        sig.iloc[pos,sig.columns.get_loc(a)]=beta(r[a].iloc[sl][shock],f[shock])-beta(r[a].iloc[sl][~shock],f[~shock])
print('CANDIDATE currency_shock_vs_quiet_beta_spread_60obs')
print('dates',len(r),'range',str(r.index.min()),str(r.index.max()),'signal_cells',int(sig.notna().sum().sum()),'/',sig.size)
# Cross-sectional Spearman IC. factor t -> returns after decision t; no same-day use.
def report(h):
    ics=[]; sizes=[]
    for p in range(60,len(r)-h):
        x=sig.iloc[p]; y=(1+r.iloc[p+1:p+1+h]).prod()-1
        ok=x.notna()&y.notna()
        if ok.sum()>=8:
            ics.append(x[ok].corr(y[ok],method='spearman'));sizes.append(int(ok.sum()))
    z=np.array(ics); mean=z.mean(); sd=z.std(ddof=1); ir=mean/sd if sd else np.nan
    print('H',h,'IC',round(mean,6),'ICIR',round(ir,6),'hit',round((z>0).mean(),4),'dates',len(z),'meanN',round(np.mean(sizes),2))
    # chronological three-regime print
    for name,sub in [('early',z[:len(z)//3]),('middle',z[len(z)//3:2*len(z)//3]),('recent',z[2*len(z)//3:])]:
        print(' ',name,'IC',round(sub.mean(),6),'ICIR',round(sub.mean()/sub.std(ddof=1),6),'N',len(sub))
for h in (1,5,10,20):report(h)
# rank turnover daily among available signal dates
v=[]
for p in range(61,len(sig)):
 x=sig.iloc[p-1];y=sig.iloc[p];ok=x.notna()&y.notna()
 if ok.sum()>=8:v.append((x[ok].rank(pct=True)-y[ok].rank(pct=True)).abs().mean())
print('turnover',round(float(np.mean(v)),6),'comparisons',len(v),'coverage',round(float(sig.notna().mean().mean()),4))
"""
