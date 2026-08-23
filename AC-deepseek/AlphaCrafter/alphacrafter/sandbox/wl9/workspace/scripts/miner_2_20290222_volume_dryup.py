"""miner_2 2029-02-22: validate volume_z_60d (attention/neglect) factor.
Candidate: z = (log(volume) - rolling_mean(log(volume),60)) / rolling_std(log(volume),60)
Motivation: relative trading-activity (attention) predicts cross-asset short-term drift.
Admission gates (15-instrument universe): |IC| >= 0.0070, |ICIR| >= 0.0840 at 10d horizon.
No future data: factor values computed with info up to t-1; forward return t..t+h-1.
"""
import pandas as pd, numpy as np, json
from pathlib import Path

CUTOFF = pd.Timestamp('2029-02-21')
TRADABLE = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E',
            'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def load(a, cutoff):
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv', parse_dates=['date'])
    df = df[df['date'] <= cutoff].set_index('date').sort_index()
    for c in ['close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

closes, vols = {}, {}
for a in TRADABLE:
    df = load(a, CUTOFF)
    closes[a] = df['close']
    vols[a] = df['volume'].replace(0, np.nan)

C = pd.DataFrame(closes)
V = pd.DataFrame(vols)
print(f'Close panel {C.shape}, volume panel {V.shape}')
print(f'Volume coverage per asset:'); print((V.notna().sum()).to_string())

# Factor: 60d z-score of log volume, shifted by 1 day (signal known before t)
W = 60
logv = np.log(V)
mu = logv.rolling(W).mean()
sd = logv.rolling(W).std()
z = (logv - mu) / sd
sig = z.shift(1)

# forward returns over horizons 1..20 (t..t+h-1)
fwd = {}
for h in [1, 5, 10, 15, 20]:
    fwd[h] = C.shift(-(h-1)) / C - 1.0  # return from t to t+h-1

def ic_stats(sig, ret, min_dates=60, min_assets=8):
    out = []
    common = sig.join(ret, lsuffix='_s', rsuffix='_r')
    # align: both must be non-nan
    dates = []
    ics = []
    for dt, row in sig.iterrows():
        s = row.dropna()
        r = ret.loc[dt].dropna()
        common_idx = s.index.intersection(r.index)
        if len(common_idx) < min_assets:
            continue
        ic = np.corrcoef(s[common_idx], r[common_idx])[0, 1]
        if np.isnan(ic):
            continue
        dates.append(dt); ics.append(ic)
    ics = np.array(ics)
    dates = pd.DatetimeIndex(dates)
    ic_mean = ics.mean()
    ic_std = ics.std(ddof=1) if len(ics) > 1 else np.nan
    icir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) and ic_std > 0 else np.nan
    hit = (np.sign(ics) == np.sign(ic_mean)).mean()
    return dict(n=len(ics), first=dates.min(), last=dates.max(), ic=ic_mean, icir=icir, hit=hit)

print('\n=== Factor: vol_z_60d (z of log volume), horizon scan (raw signal) ===')
horizons = {}
for h in [1, 5, 10, 15, 20]:
    st = ic_stats(sig, fwd[h])
    horizons[h] = st
    print(f'h={h:2d}: n={st["n"]:5d} ic={st["ic"]:+.4f} icir={st["icir"]:+.4f} hit={st["hit"]:.3f}')

h0 = 10
st10 = horizons[h0]
print(f'\nGate check @10d: |IC|={abs(st10["ic"]):.4f} (need>=0.0070), |ICIR|={abs(st10["icir"]):.4f} (need>=0.0840)')

# Both sign conventions
print('\n=== Sign conventions @10d ===')
for sign, name in [(1, 'raw (high vol z -> high ret)'), (-1, 'INVERTED (low vol z -> high ret)')]:
    s = sign * sig
    st = ic_stats(s, fwd[h0])
    print(f'{name}: ic={st["ic"]:+.4f} icir={st["icir"]:+.4f} hit={st["hit"]:.3f}')

# Robustness across regimes
print('\n=== Regime robustness @10d (raw sign, 2y rolling blocks) ===')
for lo, hi in [('2020-01-01', '2022-12-31'), ('2023-01-01', '2025-12-31'),
               ('2026-01-01', '2027-12-31'), ('2028-01-01', CUTOFF), ('2026-07-16', CUTOFF)]:
    sub = sig[(sig.index >= lo) & (sig.index <= hi)]
    st = ic_stats(sub, fwd[h0])
    print(f'{lo}..{hi}: n={st["n"]:5d} ic={st["ic"]:+.4f} icir={st["icir"]:+.4f} hit={st["hit"]:.3f}')

# Coverage & turnover
valid_mask = sig.notna()
cov_date = (valid_mask.sum(axis=1) >= 8).mean()
print(f'\nCoverage: dates with >=8 valid assets = {cov_date:.3f}; asset-day coverage = {valid_mask.mean():.3f}')
sig_use = sig.copy()
ranks = sig_use.rank(axis=1, pct=True)
turn = ranks.diff().abs().mean().mean()
print(f'Turnover (mean abs rank change/day): {turn:.4f}')

# Persist signal artifact for the gate (inverted sign if chosen later)
print('\nSaving raw signal panel for audit...')
sig.to_csv('scripts/miner_2_20290222_volz_signal_raw.csv')
print('saved scripts/miner_2_20290222_volz_signal_raw.csv', sig.shape)