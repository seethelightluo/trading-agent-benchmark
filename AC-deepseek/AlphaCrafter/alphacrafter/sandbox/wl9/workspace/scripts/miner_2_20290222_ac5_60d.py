"""miner_2 2029-02-22: validate ac5_60d (lag-5 weekly-echo autocorrelation of returns).
Candidate: ac5 = rolling correlation(r_t, r_{t-5}, 60d) -> measures whether weekly returns echo.
Motivation: US-equity cross-asset spillovers often operate at 5-day (weekly) cycle; persistence
of weekly returns differs from same-day autocorrelation (ac1_120d already in library).
Admission gates (15-asset universe): |IC| >= 0.0070, |ICIR| >= 0.0840 @10d.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

CUTOFF = pd.Timestamp('2029-02-21')
TRADABLE = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E',
            'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

C = pd.DataFrame({a: pd.read_csv(f'../persistent/stock_data/{a}.csv', parse_dates=['date'])
                  .set_index('date').sort_index()['close'].astype(float)
                  for a in TRADABLE})
C = C[C.index <= CUTOFF]
print('Close panel', C.shape, C.index.min().date(), '->', C.index.max().date())

r = C.pct_change()

def roll_corr(x, lag, win):
    return x.rolling(win).corr(x.shift(lag))

ac5 = r.apply(roll_corr, lag=5, win=60)
sig = ac5.shift(1)

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
    m = ics.mean(); sd = ics.std(ddof=1) if len(ics) > 1 else np.nan
    icir = m / sd if sd and sd == sd and sd > 0 else np.nan
    hit = (np.sign(ics) == np.sign(m)).mean() if len(ics) else np.nan
    return dict(n=len(ics), first=dates.min() if len(dates) else None,
                last=dates.max() if len(dates) else None, ic=m, icir=icir, hit=hit)

print('\n=== Factor: ac5_60d, horizon scan ===')
horizons = {}
for h in [1, 5, 10, 15, 20]:
    st = ic_stats(sig, fwd[h]); horizons[h] = st
    print(f'h={h:2d}: n={st["n"]:5d} ic={st["ic"]:+.4f} icir={st["icir"]:+.4f} hit={st["hit"]:.3f}')

st10 = horizons[10]
print(f'\nGate @10d: |IC|={abs(st10["ic"]):.4f} (>=0.0070), |ICIR|={abs(st10["icir"]):.4f} (>=0.0840)')

print('\n=== Sign conventions @10d ===')
for sign, name in [(1, 'raw (weekly echo -> high ret)'), (-1, 'INVERTED (anti-echo -> high ret)')]:
    s = sign * sig
    st = ic_stats(s, fwd[10])
    print(f'{name}: ic={st["ic"]:+.4f} icir={st["icir"]:+.4f} hit={st["hit"]:.3f}')

print('\n=== Regime robustness @10d (raw sign) ===')
for lo, hi in [('2020-01-01', '2022-12-31'), ('2023-01-01', '2025-12-31'),
               ('2026-01-01', '2027-12-31'), ('2028-01-01', CUTOFF), ('2026-07-16', CUTOFF)]:
    sub = sig[(sig.index >= lo) & (sig.index <= hi)]
    st = ic_stats(sub, fwd[10])
    print(f'{lo}..{hi}: n={st["n"]:5d} ic={st["ic"]:+.4f} icir={st["icir"]:+.4f} hit={st["hit"]:.3f}')

valid_mask = sig.notna()
cov_date = (valid_mask.sum(axis=1) >= 8).mean()
cov_ad = float(valid_mask.mean().mean())
print(f'\nCoverage: dates>=8 assets={cov_date:.3f}; asset-day={cov_ad:.3f}')
ranks = sig.rank(axis=1, pct=True)
turn = float(ranks.diff().abs().mean().mean())
print(f'Turnover (mean abs rank change/day): {turn:.4f}')

sig.to_csv('scripts/miner_2_20290222_ac5_signal_raw.csv')
print('saved signal panel', sig.shape)