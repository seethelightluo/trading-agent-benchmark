import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
TODAY='2029-12-31'; U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
try: U=get_account_dict().get('watch_list',[]) or U
except Exception: pass
px={}
for s in U:
    try: d=get_index_daily_data(s,days=4000)
    except Exception: d=None
    if d is None or len(d)<150:
        try: d=get_stock_daily_data(s,days=4000)
        except Exception: d=None
    if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]; R=P.pct_change()
# Relative residual reversal: recent asset return versus contemporaneous cross-asset median,
# but only when the idiosyncratic move is large relative to its own 20d residual risk.
rows=[]; H=10
for i in range(80,len(P)-H):
    date=P.index[i]; vals=[]
    cross=R.iloc[i-9:i].median(axis=1,skipna=True)
    for s in U:
        if s not in P: continue
        x=P[s].iloc[:i+1].dropna()
        if len(x)<61 or pd.isna(P[s].iloc[i+H]): continue
        rr=x.pct_change().dropna(); common=cross.reindex(rr.index).fillna(0)
        resid=(rr-common).dropna()
        if len(resid)<25: continue
        r10=x.iloc[-1]/x.iloc[-11]-1
        cr10=common.iloc[-10:].sum()
        rs=r10-cr10
        vol=resid.iloc[-21:-1].std()
        sig=-rs/(vol*np.sqrt(10)) if np.isfinite(vol) and vol>0 else np.nan
        fwd=P[s].iloc[i+H]/x.iloc[-1]-1
        if np.isfinite(sig) and np.isfinite(fwd): vals.append((s,sig,fwd))
    if len(vals)>=8: rows += [(date,)+v for v in vals]
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
ic=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna()
ranks=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
turn=ranks.diff().abs().mean(axis=1).dropna().mean()
print('dates',len(ic),'avg_names',df.groupby('date').size().mean(),'symbols',df.symbol.nunique(),'coverage',df.symbol.nunique()/len(U),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turn',turn)
for j,(a,b) in enumerate([(0,len(ic)//3),(len(ic)//3,2*len(ic)//3),(2*len(ic)//3,len(ic))]):
 q=ic.iloc[a:b]; print('regime',j,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if q.std()>0 else np.nan)
df.to_csv('scripts/miner_1_20291231_relative_residual_reversal_signal.csv',index=False)
