"""
Explore cross-sectional relative strength factor: rel_str_20d
Computes per-asset return over lookback, then z-scores cross-sectionally.
Captures rotation dynamics in multi-asset universe.
"""
import sys, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acct = get_account_dict()
watch_list = acct.get("watch_list", [])
print(f"Watchlist: {watch_list} ({len(watch_list)} assets)")

N_DAYS = 300
all_data = {}
for sym in watch_list:
    df = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df is not None and len(df) > 30:
        all_data[sym] = df
print(f"Loaded {len(all_data)} instruments")

H = 10

for lookback in [10, 20, 40]:
    label = f'{lookback}d'
    print(f"\n=== Relative Strength {label} (LB={lookback}, H={H}) ===")
    
    # Panel: date, symbol, ret(LB), fwd_ret(H)
    panel_rows = []
    for sym, df in all_data.items():
        cp = df.copy()
        cp['ret'] = cp['close'].pct_change(lookback)
        cp = cp.dropna()
        for i in range(len(cp) - H):
            row = cp.iloc[i]
            fwd = cp.iloc[i+H]
            panel_rows.append({'date':row['date'],'symbol':sym,
                               'ret':row['ret'],'fwd_ret':fwd['close']/row['close']-1})
    panel = pd.DataFrame(panel_rows)
    
    # Cross-sectional z-score
    fvals = []
    for dt, grp in panel.groupby('date'):
        if len(grp) >= 8:
            mu, s = grp['ret'].mean(), grp['ret'].std()
            for _, r in grp.iterrows():
                fvals.append({'date':dt,'symbol':r['symbol'],
                              'factor':(r['ret']-mu)/max(s,1e-10),'fwd_ret':r['fwd_ret']})
    fp = pd.DataFrame(fvals)
    if len(fp)==0: print("No valid fvals"); continue
    
    ics = []
    for dt, grp in fp.groupby('date'):
        if len(grp)>=8:
            c = grp['factor'].corr(grp['fwd_ret'])
            if not np.isnan(c): ics.append(c)
    ic_s = pd.Series(ics)
    if len(ic_s)==0: print("No ICs"); continue
    
    abs_ic = abs(ic_s.mean())
    icir = ic_s.mean()/max(ic_s.std(),1e-10)*np.sqrt(len(ic_s))
    abs_icir = abs(icir)
    print(f"IC obs={len(ic_s)} Mean IC={ic_s.mean():.6f} Std={ic_s.std():.6f}")
    print(f"ICIR={icir:.6f} IC>0:{(ic_s>0).sum()}/{len(ic_s)} ({(ic_s>0).mean()*100:.1f}%)")
    print(f"abs(IC)={abs_ic:.6f} {'PASS' if abs_ic>=0.0070 else 'FAIL'}")
    print(f"abs(ICIR)={abs_icir:.6f} {'PASS' if abs_icir>=0.0840 else 'FAIL'}")
    
    # Decay
    for h in [1,2,3,5,10,20]:
        hr = []
        for sym, df in all_data.items():
            cp = df.copy()
            cp['ret'] = cp['close'].pct_change(lookback)
            cp = cp.dropna()
            for i in range(len(cp)-h):
                fwd = cp.iloc[i+h]
                hr.append({'date':cp.iloc[i]['date'],'symbol':sym,
                           'ret':cp.iloc[i]['ret'],'fwd_ret':fwd['close']/cp.iloc[i]['close']-1})
        hp = pd.DataFrame(hr)
        hf = []
        for dt, grp in hp.groupby('date'):
            if len(grp)>=8:
                mu,s = grp['ret'].mean(),grp['ret'].std()
                for _,r in grp.iterrows():
                    hf.append({'date':dt,'factor':(r['ret']-mu)/max(s,1e-10),'fwd_ret':r['fwd_ret']})
        hfp = pd.DataFrame(hf)
        hics = []
        for dt,g in hfp.groupby('date'):
            if len(g)>=8:
                c = g['factor'].corr(g['fwd_ret'])
                if not np.isnan(c): hics.append(c)
        if hics:
            print(f"  H={h:2d}: IC={np.mean(hics):.6f} ICIR={np.mean(hics)/max(np.std(hics),1e-10)*np.sqrt(len(hics)):.6f}")