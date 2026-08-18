"""miner_2 cycle 2026-12-31: trend/momentum family screen (idea #1).
Live-aware: 000688.SH/CN10Y/NDX/SOX frozen since 2026-07-16; report IC both on
full 15-asset panel and live-only panel.
Admission gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at H=10 on 15-asset cross-section.
"""
import sys, os, json, zlib, base64, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, cross_sectional_ic,
                                     ic_stats, spearman_panel_rho)

ASOF = '2026-12-30'
H = 10
px = load_prices(ASOF)
fwd = px.shift(-H) / px - 1.0

DEAD = ['000688.SH', 'CN10Y', 'NDX', 'SOX']
LIVE = [s for s in WATCH if s not in DEAD]
print(f'ASOF={ASOF} n_dates_total={len(px)} live={len(LIVE)} dead={DEAD}')

def ret(s, k, skip=0):
    return px[s] / px[s].shift(k + skip) - 1.0

def eff_ratio(s, w):
    r = px[s].diff()
    path = r.abs().rolling(w).sum()
    net = (px[s] - px[s].shift(w)).abs()
    return net / path

def build(fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s)
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()

F = {}

# F1 baseline: mom_10d_skip5 (recompute for freshness)
F['mom_10d_skip5'] = build(lambda s: ret(s, 10, skip=5))
# F2 mom_5d_skip2
F['mom_5d_skip2'] = build(lambda s: ret(s, 5, skip=2))
# F3 mom_20d_skip5
F['mom_20d_skip5'] = build(lambda s: ret(s, 20, skip=5))
# F4 mom_40d_skip5
F['mom_40d_skip5'] = build(lambda s: ret(s, 40, skip=5))
# F5 mom_60d_skip5
F['mom_60d_skip5'] = build(lambda s: ret(s, 60, skip=5))
# F6 acceleration: mom_20d_skip5 - mom_60d_skip5
F['accel_20v60'] = build(lambda s: ret(s, 20, skip=5) - ret(s, 60, skip=5))
# F7 acceleration: mom_10d_skip5 - mom_40d_skip5
F['accel_10v40'] = build(lambda s: ret(s, 10, skip=5) - ret(s, 40, skip=5))
# F8 trend-quality: mom_20d_skip5 * eff_ratio_20
F['trend_quality_20'] = build(lambda s: ret(s, 20, skip=5) * eff_ratio(s, 20))
# F9 hl_pos_60: (close - min60)/(max60-min60) - 0.5 (negative expected: buy lows)
F['hl_pos_60'] = build(lambda s: (px[s] - px[s].rolling(60).min()) /
                                (px[s].rolling(60).max() - px[s].rolling(60).min()) - 0.5)
# F10 range_pos_20
F['range_pos_20'] = build(lambda s: (px[s] - px[s].rolling(20).min()) /
                                   (px[s].rolling(20).max() - px[s].rolling(20).min()) - 0.5)
# F11 time-weighted momentum 40d (exponential weights, half-life ~10d)
def tw_mom(s, w=40, halflife=10):
    lam = np.log(2) / halflife
    wts = np.exp(-lam * np.arange(w, 0, -1))
    wts /= wts.sum()
    r = px[s].pct_change()
    return r.rolling(w).apply(lambda x: float(np.dot(wts, x)), raw=True)
F['tw_mom_40'] = build(lambda s: tw_mom(s))

# library panels
def decode_usdcny():
    d = json.load(open('factors/usdcny_beta_60.json'))
    csv_txt = zlib.decompress(base64.b64decode(d['validation']['signal_artifact']['data'])).decode()
    df = pd.read_csv(io.StringIO(csv_txt))
    return df.set_index(df.columns[0])

vix = pd.read_csv('../persistent/index_data/VIX.csv'); vix['date'] = pd.to_datetime(vix['date'])
vix = vix[vix['date'] <= pd.Timestamp(ASOF)].set_index('date')['close']

lib = {'usdcny_beta_60': decode_usdcny()}
lib['mom_10d_skip5_recalc'] = F['mom_10d_skip5']
def vix_beta_cond():
    vchg20 = (vix / vix.shift(20) - 1.0) > 0
    out = {}
    for s in WATCH:
        r = px[s].pct_change()
        cond = vchg20.reindex(px.index).fillna(False)
        cov = r.where(cond).rolling(60, min_periods=24).cov(vix.pct_change().where(cond))
        var = vix.pct_change().where(cond).rolling(60, min_periods=24).var()
        out[s] = cov / var
    return pd.DataFrame(out).sort_index().replace([np.inf, -np.inf], np.nan)
lib['vix_beta_cond_60x20'] = vix_beta_cond()

def yield_beta_cond():
    us10y = px['US10Y']
    ychg20 = (us10y / us10y.shift(20) - 1.0) > 0
    out = {}
    for s in WATCH:
        r = px[s].pct_change()
        cond = ychg20.reindex(px.index).fillna(False)
        cov = r.where(cond).rolling(60, min_periods=24).cov(us10y.pct_change().where(cond))
        var = us10y.pct_change().where(cond).rolling(60, min_periods=24).var()
        out[s] = cov / var
    return pd.DataFrame(out).sort_index().replace([np.inf, -np.inf], np.nan)
lib['yield_beta_cond_60x20'] = yield_beta_cond()

def turnover_10d(f):
    ranks = f.rank(axis=1)
    valid = ranks.notna()
    rr = ranks.fillna(-99)
    chg = (rr.shift(-10) != rr).astype(float).where(valid)
    return chg.mean().mean()

print(f'\n{"factor":20s} {"IC":>8s} {"ICIR":>8s} {"hit":>6s} {"n":>5s} {"cov":>6s} {"TO10":>6s} {"liveIC":>8s} {"liveN":>5s} {"6mIC":>7s} {"3mIC":>7s} {"maxRho":>7s}')
for name, f in F.items():
    icdf = cross_sectional_ic(f, fwd)
    st = ic_stats(icdf)
    f6 = f[f.index >= '2026-07-01']; ic6 = ic_stats(cross_sectional_ic(f6, fwd.reindex(f6.index)))
    f3 = f[f.index >= '2026-10-01']; ic3 = ic_stats(cross_sectional_ic(f3, fwd.reindex(f3.index)))
    # live-only IC
    fl = f[LIVE]; fwdl = fwd[LIVE]
    icl = cross_sectional_ic(fl, fwdl); stl = ic_stats(icl)
    # library corr
    rhos = []
    for ln, lp in lib.items():
        r = spearman_panel_rho(f, lp)
        if not np.isnan(r):
            rhos.append(abs(r))
    mx = max(rhos) if rhos else np.nan
    to = turnover_10d(f)
    cov = f.notna().mean().mean()
    print(f'{name:20s} {st["ic"]:8.4f} {st["icir"]:8.4f} {st["hit"]:6.3f} {st["n_dates"]:5d} '
          f'{cov:6.3f} {to:6.3f} {stl["ic"]:8.4f} {stl["n_dates"]:5d} '
          f'{ic6["ic"]:7.4f} {ic3["ic"]:7.4f} {mx:7.4f}')

# regime snapshot of live assets
print('\n--- recent 1M/3M returns (live assets) ---')
for s in LIVE:
    p = px[s]
    r1 = p.iloc[-1] / p[p.index <= p.index[-1] - pd.Timedelta(days=31)].iloc[-1] - 1
    r3 = p.iloc[-1] / p[p.index <= p.index[-1] - pd.Timedelta(days=92)].iloc[-1] - 1
    print(f'{s:10s} 1M={r1*100:7.2f}% 3M={r3*100:7.2f}% last={p.iloc[-1]:.3f}')