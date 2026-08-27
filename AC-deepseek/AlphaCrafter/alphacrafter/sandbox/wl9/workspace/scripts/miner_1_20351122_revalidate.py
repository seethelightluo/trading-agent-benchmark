"""Re-validate ensemble factors on most recent data"""
import pandas as pd
import numpy as np
from pathlib import Path

STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUR_DATE = '2035-11-22'

closes = {}
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= CUR_DATE]
    closes[a] = df.set_index('date')['close'].astype(float)

rets = pd.DataFrame({a: closes[a].pct_change() for a in ASSETS}).dropna()
rets = rets[rets.index >= '2020-03-01']

vix = pd.read_csv(INDEX_DIR / 'VIX.csv', parse_dates=['date'])
vix = vix[vix['date'] <= CUR_DATE].set_index('date')['close'].astype(float)

macros = {}
for m in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    f = INDEX_DIR / f'{m}.csv'
    df = pd.read_csv(f, parse_dates=['date'])
    df = df[df['date'] <= CUR_DATE].set_index('date')['close'].astype(float)
    macros[m] = df

fwd_10d = rets.shift(-10).rolling(10).mean()
vix_rets = vix.pct_change()

def compute_ic(factor_vals, forward_rets):
    common_dates = sorted(set(factor_vals.index) & set(forward_rets.index))
    ics = []
    for d in common_dates:
        f = factor_vals.loc[d]
        r = forward_rets.loc[d]
        valid = f.notna() & r.notna()
        if valid.sum() >= 8:
            fv = f[valid].rank().values
            rv = r[valid].rank().values
            if np.std(fv) > 0 and np.std(rv) > 0:
                ic = np.corrcoef(fv, rv)[0,1]
                ics.append(ic)
    ics = np.array(ics)
    if len(ics) < 20:
        return {'IC': 0.0, 'ICIR': 0.0, 'n_dates': len(ics)}
    return {
        'IC': float(np.mean(ics)),
        'ICIR': float(np.mean(ics) / np.std(ics) * np.sqrt(len(ics))) if np.std(ics) > 0 else 0.0,
        'n_dates': len(ics)
    }

def compute_factor_vals(fn, label):
    """Compute factor values using the provided function, IC, and return vals"""
    # For kaufman, vol_z, bb_width, we need period-specific checks
    pass

# ===== Period-based revalidation =====
for label, cutoff in [('RECENT_2029', '2029-01-01'), ('RECENT_2031', '2031-01-01'), ('RECENT_2033', '2033-01-01')]:
    print(f"\n{'='*60}")
    print(f"  {label} (from {cutoff})")
    print(f"{'='*60}")
    srets = rets[rets.index >= cutoff]
    sfwd = fwd_10d.reindex(srets.index)
    dxy = macros['DXY'].pct_change()
    
    # 1. beta_VIX_60
    b60 = pd.DataFrame(index=srets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        j = pd.concat([srets[a].rename('a'), vix_rets.reindex(srets.index).rename('v')], axis=1).dropna()
        b60[a] = j['a'].rolling(60).cov(j['v']) / j['v'].rolling(60).var()
    ic = compute_ic(b60, sfwd)
    print(f"  beta_VIX_60: IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")

    # 2. kaufman_eff_20d
    kf = pd.DataFrame(index=srets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        c = closes[a]
        kf[a] = (c.diff(20).abs() / c.diff().abs().rolling(20).sum()).reindex(srets.index)
    ic = compute_ic(kf, sfwd)
    print(f"  kaufman_eff_20d: IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")

    # 3. mom_120d_skip5
    mom120 = pd.DataFrame(index=srets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        mom120[a] = closes[a].pct_change(125).reindex(srets.index)
    ic = compute_ic(mom120, sfwd)
    print(f"  mom_120d_skip5: IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")

    # 4. bb_width_20d
    bb = pd.DataFrame(index=srets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        c = closes[a]
        bb[a] = ((c - c.rolling(20).mean()) / c.rolling(20).std()).reindex(srets.index)
    ic = compute_ic(bb, sfwd)
    print(f"  bb_width_20d: IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")

    # 5. vol_z_20d
    vol = pd.DataFrame(index=srets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        v = srets[a].rolling(20).std()
        vol[a] = (v - v.rolling(60).mean()) / v.rolling(60).std()
    ic = compute_ic(vol, sfwd)
    print(f"  vol_z_20d: IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")

    # 6. cny_beta_60 - beta of asset on USDCNY 60d
    usdcny = macros['USDCNY'].pct_change()
    cn60 = pd.DataFrame(index=srets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        j = pd.concat([srets[a].rename('a'), usdcny.reindex(srets.index).rename('c')], axis=1).dropna()
        cn60[a] = j['a'].rolling(60).cov(j['c']) / j['c'].rolling(60).var()
    ic = compute_ic(cn60, sfwd)
    print(f"  cny_beta_60: IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")

    # 7. ac1_120d
    ac1 = pd.DataFrame(index=srets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        ac1[a] = srets[a].rolling(120).apply(lambda x: x.autocorr(1) if len(x)>=30 else np.nan, raw=False)
    ic = compute_ic(ac1, sfwd)
    print(f"  ac1_120d: IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")

    # 8. mom_10_vixreg
    mom10raw = pd.DataFrame(index=srets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        mom10raw[a] = closes[a].pct_change(15).reindex(srets.index)
    vix_z = (vix.reindex(srets.index) - vix.reindex(srets.index).rolling(60).mean()) / vix.reindex(srets.index).rolling(60).std()
    mom10_vr = pd.DataFrame(index=srets.index, columns=ASSETS, dtype=float)
    for