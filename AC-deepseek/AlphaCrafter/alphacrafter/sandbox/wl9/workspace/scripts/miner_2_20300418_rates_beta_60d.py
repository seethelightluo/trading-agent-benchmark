"""miner_2 2030-04-18: validate rates_beta_60d.
Candidate: beta of each asset's daily return to US10Y-yield change over 60d.
Motivation: library already has beta factors for VIX, DXY, CNY (macro regime).
Rate beta identifies rate-gated assets; high rate-beta should underperform when
rates rising. Complementary macro factor.
No future data: windows use info up to t-1; forward returns t..t+h-1.
Admission gates (15-asset universe): |IC| >= 0.0070, |ICIR| >= 0.0840 @10d.
"""
import pandas as pd, numpy as np, json, base64, zlib, io, hashlib
from pathlib import Path

CUTOFF = pd.Timestamp('2030-04-17')
TRADABLE = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E',
            'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
SD = Path('../persistent/stock_data')

def load(a, cutoff):
    df = pd.read_csv(SD / f'{a}.csv', parse_dates=['date'])
    df = df[df['date'] <= cutoff].set_index('date').sort_index()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df

C = pd.DataFrame({a: load(a, CUTOFF)['close'] for a in TRAD})
R = C.pct_change()

# US10Y level is the asset's "close" series
us10y_lvl = C['US10Y']
d_rates = us10y_lvl.pct_change()  # relative yield change

W = 60
# rolling beta: cov(asset_ret, dRates, W)/var(dRates, W)
def rolling_beta(asset_ret, mkt, win):
    cov = asset_ret.rolling(win).cov(mkt)
    var = mkt.rolling(win).var()
    return (cov / var).where(var > 1e-14)

beta = pd.DataFrame({a: rolling_beta(R[a], d_rates, W) for a in TRAD})
sig = beta.shift(1)  # signal known at t-1 using data through t-1

fwd = {h: C.shift(-(h-1)) / C - 1.0 for h in [1, 5, 10, 15, 20]}

def ic_stats(sig, ret, min_assets=8):
    dates, ics = [], []
    for dt, row in sig.iterrows():
        s = row.dropna(); rr = ret.loc[dt].dropna()
        idx = s.index.intersection(rr.index)
        if len(idx) < min_assets: continue
        ic = np.corrcoef(s[idx], rr[idx])[0, 1]
        if not np.isnan(ic): dates.append(dt); ics.append(ic)
    ics = np.array(ics); dates = pd.DatetimeIndex(dates)
    m = ics.mean()
    sd = ics.std(ddof=1) if len(ics) > 1 else np.nan
    icir = m / sd if sd and sd == sd and sd > 0 else np.nan
    hit = (np.sign(ics) == np.sign(m)).mean() if len(ics) else np.nan
    return dict(n=len(ics), first=dates.min() if len(dates) else None,
                last=dates.max() if len(dates) else None, ic=m, icir=icir, hit=hit)

print('=== Factor: rates_beta_60d (US10Y rate-beta), horizon scan ===')
hors = {}
for h in [1, 5, 10, 15, 20]:
    st = ic_stats(sig, fwd[h]); hors[h] = st
    print(f'h={h:2d}: n={st["n"]:5d} ic={st["ic"]:+.4f} icir={st["icir"]:+.4f} hit={st["hit"]:.3f}')

st10 = hors[10]
print(f'\nGate @10d: |IC|={abs(st10["ic"]):.4f} (>=0.0070), |ICIR|={abs(st10["icir"]):.4f} (>=0.0840)')

# also test INVERTED sign
s_neg = -sig
stn = ic_stats(s_neg, fwd[10])
print(f'INVERTED: n={stn["n"]} ic={stn["ic"]:+.4f} icir={stn["icir"]:+.4f} hit={stn["hit"]:.3f}')

print('\n=== Regime robustness @10d (adopt best sign) ===')
best = sig if abs(st10['ic']) >= abs(stn['ic']) else -sig
for lo, hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),
               ('2026-01-01','2027-12-31'),('2028-01-01',CUTOFF),
               ('2026-07-16',CUTOFF),('2029-01-01','2029-12-31'),
               ('2029-09-01','2030-03-31')]:
    sub = best[(best.index >= lo) & (best.index <= hi)]
    st = ic_stats(sub, fwd[10])
    print(f'{lo}..{hi}: n={st["n"]:5d} ic={st["ic"]:+.4f} icir={st["icir"]:+.4f} hit={st["hit"]:.3f}')

valid_mask = sig.notna()
cov_date = (valid_mask.sum(axis=1) >= 8).mean()
cov_ad = float(valid_mask.mean().mean())
ranks = sig.rank(axis=1, pct=True)
turn = float(ranks.diff().abs().mean().mean())
print(f'\nCoverage: dates>=8={cov_date:.3f}; asset-day={cov_ad:.3f}; turnover={turn:.4f}')

# max_abs_library_correlation vs existing beta factors (provenance)
print('\n=== max_abs_library_correlation vs beta_VIX_60 / cny_beta_60 ===')
try:
    lib = {}
    for fid in ['beta_VIX_60', 'cny_beta_60']:
        d = json.load(open(f'factors/{fid}.json'))
        art = d['validation']['signal_artifact']
        raw = zlib.decompress(base64.b64decode(art['data']))
        s = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
        s.columns = TRAD
        s = s.reindex(sig.index)
        lib[fid] = s
    rho_list = []
    for dt in sig.index:
        base_s = sig.loc[dt].dropna()
        if len(base_s) < 8: continue
        for fid, s in lib.items():
            o = s.loc[dt] if dt in s.index else None
            if o is None: continue
            both = base_s.index.intersection(o.dropna().index)
            if len(both) < 8: continue
            r = np.corrcoef(base_s[both], o[both])[0,1]
            if not np.isnan(r): rho_list.append(r)
    rho_list = np.array(rho_list)
    maxabs = np.max(np.abs(rho_list)) if len(rho_list) else 0.0
    meanabs = np.mean(np.abs(rho_list)) if len(rho_list) else 0.0
    print(f'n_pairs={len(rho_list)} mean_abs_rho={meanabs:.4f} max_abs_rho={maxabs:.4f}')
except Exception as e:
    print('lib correlation not computed:', repr(e))
    maxabs = None