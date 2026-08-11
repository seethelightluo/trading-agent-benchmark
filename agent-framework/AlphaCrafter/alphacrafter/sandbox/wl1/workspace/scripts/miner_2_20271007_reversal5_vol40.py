import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def fetch(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=3000)
            if x is not None and len(x)>100: return x
        except Exception: pass
    return None

def main():
    ds={s:fetch(s) for s in U}; ds={s:x for s,x in ds.items() if x is not None}
    print('symbols',len(ds), 'lengths', {s:len(x) for s,x in ds.items()})
    close=pd.DataFrame({s:x.set_index('date')['close'] for s,x in ds.items()}).sort_index()
    ret=close.pct_change()
    # candidate: short reversal, smoothed by 5d return, 40d vol, lagged one completed session
    raw=-(close/close.shift(5)-1)/(ret.rolling(40).std()*np.sqrt(252)+0.005)
    fac=raw.shift(1)
    results=[]
    for h in [5,10,20]:
        fwd=close.shift(-h)/close-1
        vals=[]
        for d in fac.index:
            z=pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna()
            if len(z)>=8: vals.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
        a=pd.DataFrame(vals,columns=['date','ic','n'])
        if len(a):
            print('h',h,'dates',len(a),'avgN',a.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1), (a.ic>0).mean()))
            print('regimes',[(yr, round(a[a.date.dt.year==yr].ic.mean(),6),len(a[a.date.dt.year==yr])) for yr in sorted(a.date.dt.year.unique())[-4:]])
        if h==20:
            # rank turnover based consecutive valid ranks
            ranks=fac.rank(axis=1,pct=True); tr=(ranks.diff().abs().mean(axis=1)).dropna().mean()
            print('coverage',fac.notna().mean().mean(),'rank_turnover',tr)
            out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20271007_reversal5_vol40_signal.csv',index=False)
if __name__=='__main__': main()
