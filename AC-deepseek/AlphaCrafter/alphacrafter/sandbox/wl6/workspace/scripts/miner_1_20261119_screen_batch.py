"""miner_1 exploration 2026-11-19: broad screen of NEW factor ideas.

Context: ensemble momentum factors decaying (mom_10d_skip5 recent ic<0), last
block -2.66%. Explore reversal / trend-quality / vol-structure / macro-beta /
idiosyncratic families to find orthogonal signals.

Method: macro-weekday calendar truncated to visible_through (2026-11-04).
Gate (shared, h=10 warm-up 2020-01-01..2026-07-15): |IC|>=0.0070, |ICIR|>=0.0840.
Also report recent-400d window for freshness and pooled rank correlation vs the
4-factor library (inline strategy.py formulas). No backtest/step calls.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'VIX', 'USDCNY', 'USDJPY', 'EURUSD']
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
VTH = pd.Timestamp('2026-11-04')
WARM_END = pd.Timestamp('2026-07-15')
H = 10
MIN_A = 8
IC_TH, ICIR_TH = 0.0070, 0.0840

# ---------------- data ----------------
frames = {}
for s in MACRO:
    df = pd.read_csv(INDEX_DIR / f'{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VTH].sort_values('date')
    frames[s] = df.set_index('date')['close'].astype(float)
macro = pd.DataFrame(frames).sort_index()
cal = macro.index
print(f'macro calendar: {len(cal)} weekdays, {cal.min().date()}..{cal.max().date()}')

closes, highs, lows = {}, {}, {}
for s in WATCH:
    df = pd.read_csv(STOCK_DIR / f'{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VTH].sort_values('date').set_index('date')
    closes[s] = df['close'].astype(float).reindex(cal)
    highs[s] = df['high'].astype(float).reindex(cal)
    lows[s] = df['low'].astype(float).reindex(cal)
P = pd.DataFrame(closes).sort_index()
Hh = pd.DataFrame(highs).sort_index()
Ll = pd.DataFrame(lows).sort_index()
print(f'close panel: {P.shape}, valid frac: {P.notna().mean().mean():.3f}')

R = P.pct_change()
mkt = R.mean(axis=1, skipna=True)
fwd10 = P.shift(-H) / P - 1.0


def ic_series(F, fwd, fmin=None, fmax=None):
    out, n_out = {}, {}
    idx = F.index[(F.index >= (fmin or F.index.min())) & (F.index <= (fmax or F.index.max()))]
    for t in idx:
        fv, rv = F.loc[t], fwd.loc[t]
        m = fv.notna() & rv.notna()
        if m.sum() >= MIN_A:
            ic = fv[m].rank().corr(rv[m].rank())
            if np.isfinite(ic):
                out[t] = ic
                n_out[t] = int(m.sum())
    return pd.Series(out), pd.Series(n_out)


def summ(name, F, label='full'):
    ics, ns = ic_series(F, fwd10)
    if len(ics) == 0:
        return dict(name=name, warm=None, recent=None, cov=float(F.notna().to_numpy().mean()),
                    turn10=float('nan'), warm_ic=np.nan, warm_icir=np.nan, rec_ic=np.nan, rec_icir=np.nan)
    warm = ics[(ics.index >= pd.Timestamp('2020-01-01')) & (ics.index <= WARM_END)]
    recent = ics.iloc[-400:] if len(ics) else pd.Series(dtype=float)
    def stat(s):
        if len(s) < 30:
            return None
        sd = s.std(ddof=1)
        return dict(ic=float(s.mean()), icir=float(s.mean() / sd) if sd > 0 else 0.0,
                    hit=float((s > 0).mean()), n=int(len(s)))
    sw, sr = stat(warm), stat(recent)
    cov = float(F.notna().to_numpy().mean())
    rk = F.rank(axis=1)
    to = float(rk.diff(10).abs().mean().mean()) if len(rk) > 11 else float('nan')
    return dict(name=name, warm=sw, recent=sr, cov=cov, turn10=to,
                warm_ic=sw['ic'] if sw else np.nan, warm_icir=sw['icir'] if sw else np.nan,
                rec_ic=sr['ic'] if sr else np.nan, rec_icir=sr['icir'] if sr else np.nan)


# ---------------- candidate signals ----------------
C = {}
s5, s25 = P.shift(5), P.shift(25)
C['rev_5d'] = -(P / P.shift(5) - 1.0)
C['rev_20d_skip5'] = -(s5 / s25 - 1.0)
C['zscore_20d'] = (P - P.rolling(20).mean()) / P.rolling(20).std()
C['dist_high_20d'] = P / P.rolling(20).max() - 1.0
C['dist_high_60d'] = P / P.rolling(60).max() - 1.0
C['dist_low_20d'] = P / P.rolling(20).min() - 1.0

lp = np.log(P)
tt = np.arange(len(P), dtype=float)
# rolling R2 and slope via covariances on log price vs time
mu_x = lp.rolling(60).mean()
mu_y = pd.Series(tt, index=P.index).rolling(60).mean()
cov_xy = (lp * pd.Series(tt, index=P.index)).rolling(60).mean() - mu_x * mu_y
var_x = lp.rolling(60).var()
var_y = pd.Series(tt, index=P.index).rolling(60).var()
C['trend_r2_60d'] = (cov_xy ** 2) / (var_x * var_y)
C['trend_slope_60d'] = cov_xy / var_y
C['sma_cross_20_60'] = P.rolling(20).mean() / P.rolling(60).mean() - 1.0
C['ts_mom120_voladj'] = (s5 / P.shift(125) - 1.0) / R.rolling(20).std()
ema12 = P.ewm(span=12, adjust=False).mean()
ema26 = P.ewm(span=26, adjust=False).mean()
C['macd_12_26'] = (ema12 - ema26) / P

v5, v60 = R.rolling(5).std(), R.rolling(60).std()
C['vol_ratio_5_60'] = v5 / v60
C['skew_60d'] = R.rolling(60).skew()
C['kurt_60d'] = R.rolling(60).kurt()
C['parkinson_20d'] = (np.log(Hh / Ll) ** 2).rolling(20).mean() / (4 * np.log(2))
C['range_ratio_20d'] = ((Hh - Ll) / P).rolling(20).mean()

beta_mkt = R.rolling(60).cov(mkt) / mkt.rolling(60).var()
res_ret = R - beta_mkt * mkt
C['beta_60d'] = beta_mkt
C['res_mom_20d'] = res_ret.rolling(20).sum()
C['idio_vol_20d'] = res_ret.rolling(20).std()

def macro_beta_factor(asset_ret, macro_ret, macro_lvl, win=60, move=20):
    b = asset_ret.rolling(win).cov(macro_ret) / macro_ret.rolling(win).var()
    m = macro_lvl / macro_lvl.shift(move) - 1.0
    return -b * m

dxy_r = macro['DXY'].pct_change()
C['dxy_beta_60x20'] = macro_beta_factor(R, dxy_r, macro['DXY'])
cny_r = macro['USDCNY'].pct_change()
C['usdcny_beta_60x20'] = macro_beta_factor(R, cny_r, macro['USDCNY'])
jpy_r = macro['USDJPY'].pct_change()
C['usdjpy_beta_60x20'] = macro_beta_factor(R, jpy_r, macro['USDJPY'])
eur_r = macro['EURUSD'].pct_change()
C['eurusd_beta_60x20'] = macro_beta_factor(R, eur_r, macro['EURUSD'])
vix_r = macro['VIX'].pct_change()
C['vix_beta_raw_60'] = R.rolling(60).cov(vix_r) / vix_r.rolling(60).var()

# ---------------- library signals (inline strategy formulas) ----------------
lib = {}
lib['mom_10d_skip5'] = s5 / P.shift(15) - 1.0
lib['mom_120d_skip5'] = s5 / P.shift(125) - 1.0
lib['vol_of_vol20x60'] = R.rolling(20).std().rolling(60).std()
lib['vix_beta_cond_60x20'] = -R.rolling(60).cov(vix_r) / vix_r.rolling(60).var() * (macro['VIX'] / macro['VIX'].shift(20) - 1.0)

# sanity: library IC on warm-up
print('\n=== LIBRARY SANITY (warm-up) ===')
for k, v in lib.items():
    s = summ(k, v)
    w = s['warm']
    print(f"  {k:22s} warm ic={s['warm_ic']:.4f} icir={s['warm_icir']:.4f} n={w['n'] if w else 0} cov={s['cov']:.3f}")


def lib_corr(F):
    best, details = 0.0, {}
    for k, sig in lib.items():
        both = pd.concat([F.stack().rename('c'), sig.stack().rename('l')], axis=1).dropna()
        if len(both) < 30:
            continue
        rho = float(both['c'].rank().corr(both['l'].rank()))
        details[k] = round(rho, 3)
        best = max(best, abs(rho))
    return best, details


# ---------------- screen ----------------
rows = []
for name, F in C.items():
    s = summ(name, F)
    rho, det = lib_corr(F)
    s['libcorr'] = rho
    s['libdet'] = det
    rows.append(s)
res = pd.DataFrame(rows).sort_values('warm_icir', key=lambda x: x.abs(), ascending=False)
print('\n=== CANDIDATE SCREEN (sorted by |warm ICIR|) ===')
print(f"{'name':22s} {'warmIC':>8s} {'warmICIR':>8s} {'warmN':>6s} {'recIC':>7s} {'recICIR':>8s} {'cov':>5s} {'turn10':>6s} {'librho':>6s}")
for _, r in res.iterrows():
    w = r['warm']
    print(f"{r['name']:22s} {r['warm_ic']:8.4f} {r['warm_icir']:8.4f} {w['n'] if w else 0:6d} "
          f"{r['rec_ic']:7.4f} {r['rec_icir']:8.4f} {r['cov']:5.3f} {r['turn10']:6.2f} {r['libcorr']:6.3f}")

print('\n=== PASS GATE (|warmIC|>=0.007 & |warmICIR|>=0.084 & libcorr<0.5) ===')
for _, r in res.iterrows():
    ok = (abs(r['warm_ic']) >= IC_TH) and (abs(r['warm_icir']) >= ICIR_TH) and (r['libcorr'] < 0.5)
    flag = 'PASS' if ok else '   -'
    print(f"  {flag} {r['name']:22s} warm ic={r['warm_ic']:.4f} icir={r['warm_icir']:.4f} rec ic={r['rec_ic']:.4f} icir={r['rec_icir']:.4f} libcorr={r['libcorr']:.3f} det={r['libdet']}")

with open('scripts/miner_1_20261119_screen_batch_result.json', 'w') as fh:
    json.dump(res.to_dict(orient='records'), fh, indent=1, default=str)
print('\nresult written')
