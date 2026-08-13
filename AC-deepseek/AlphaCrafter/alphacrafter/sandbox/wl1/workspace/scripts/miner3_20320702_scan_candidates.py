"""miner3 2032-07-02: vectorized candidate factor scan on fresh panel (through 2032-07-01).
Replaces the pandas per-date corr loop (timed out) with numpy rank IC.
"""
import pandas as pd, numpy as np, json, time

panel = pd.read_pickle('scripts/panel_cache_20320702.pkl')
close = panel['close']; ret = panel['ret']; vol = panel['vol']; macro = panel['macro']
opn = panel['open']; hi = panel['high']; lo = panel['low']
vix = macro['VIX']
print("panel:", close.shape, close.index.min().date(), "->", close.index.max().date(), flush=True)

lnc = np.log(close)

def daily_rank_ic(F, R, min_n=8):
    """F,R: (n_dates, n_assets) float arrays. Returns IC per date (vectorized numpy, no scipy)."""
    F = np.asarray(F, dtype=float); R = np.asarray(R, dtype=float)
    valid = ~(np.isnan(F) | np.isnan(R))
    n = valid.sum(axis=1)
    ics = np.full(len(F), np.nan)
    for i in range(len(F)):
        ni = n[i]
        if ni < min_n:
            continue
        f = F[i, valid[i]]; r = R[i, valid[i]]
        fr = np.argsort(np.argsort(f)) + 1.0
        rr = np.argsort(np.argsort(r)) + 1.0
        fm = fr - fr.mean(); rm = rr - rr.mean()
        denom = np.sqrt((fm * fm).sum() * (rm * rm).sum())
        ics[i] = (fm * rm).sum() / denom if denom > 0 else np.nan
    return ics

def summarize(ics):
    ics = ics[~np.isnan(ics)]
    if len(ics) == 0:
        return dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
    ic = float(ics.mean()); sd = float(ics.std(ddof=1))
    return dict(ic=ic, icir=ic/sd if sd > 0 else np.nan,
                hit=float(np.mean(ics > 0)), n=len(ics))

def eval_factor(signal, horizons=(1, 2, 3, 5, 10)):
    out = {}
    S = signal.reindex(close.index)
    out['coverage'] = float(S.notna().mean().mean())
    out['n_dates'] = int(S.shape[0])
    Sv = S.values
    for h in horizons:
        fwd = (close.shift(-h) / close - 1.0).reindex(S.index).values
        ics = daily_rank_ic(Sv, fwd)
        out[h] = summarize(ics)
    return out

# ---------------- candidate factor constructions ----------------
rv20 = ret.rolling(20).std(); rv60 = ret.rolling(60).std()

sig = {}
# 1. RSI(14) oscillator (mean reversion)
def rsi(px, n=14):
    d = px.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
sig['rsi_14d'] = rsi(close, 14)

# 2. Bollinger band position (contrarian)
for n in [20, 60]:
    ma = close.rolling(n).mean(); sd = close.rolling(n).std()
    sig[f'bb_pos_{n}d'] = (close - (ma - 2*sd)) / ((ma + 2*sd) - (ma - 2*sd)).replace(0, np.nan)

# 3. Risk-adjusted momentum (mom scaled by vol)
for n in [20, 60]:
    sig[f'vol_adj_mom_{n}d'] = (lnc - lnc.shift(n)) / rv20.replace(0, np.nan) if n == 20 else (lnc - lnc.shift(n)) / rv60.replace(0, np.nan)

# 4. Range squeeze: 5d range / 20d range (breakout anticipation, low = squeeze)
for s in [5, 10]:
    rg_s = (hi.rolling(s).max() - lo.rolling(s).min())
    rg_20 = (hi.rolling(20).max() - lo.rolling(20).min())
    sig[f'squeeze_{s}x20'] = rg_s / rg_20.replace(0, np.nan)

# 5. Return autocorrelation (trend persistence)
for n in [10, 20]:
    sig[f'autocorr_{n}d'] = ret.rolling(n).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 3 else np.nan, raw=True)

# 6. VIX-conditioned 5d reversal (high VIX amplifies reversal)
vix_lvl = vix / vix.rolling(250).mean()
sig['vix_cond_rev5'] = -((lnc - lnc.shift(5)) / rv20.replace(0, np.nan)) * vix_lvl.clip(lower=1.0)

# 7. Idiosyncratic vol: asset 20d vol / cross-sectional median vol (relative riskiness)
xs_med_vol = rv20.median(axis=1)
sig['idiosyn_vol_20d'] = rv20 / xs_med_vol.replace(0, np.nan).to_frame().values

# 8. Gap reversion: today's open gap vs prev close (negative = gap up, mean revert)
gap = lnc - np.log(opn)
sig['gap_rev_1d'] = -gap

# 9. 60d range position (distance from 60d low normalized)
hi60 = hi.rolling(60).max(); lo60 = lo.rolling(60).min()
sig['range_pos_60d'] = (close - lo60) / (hi60 - lo60).replace(0, np.nan)

# 10. Drawdown recovery speed: 20d return / distance below 120d high
run_hi = close.rolling(120).max()
sig['dd_recovery_20d'] = (close / run_hi) * (close / close.shift(20) - 1.0)

# 11. Volume-weighted momentum (price x volume confirmation)
sig['vol_wtd_mom_20d'] = (lnc - lnc.shift(20)) * (vol / vol.rolling(20).mean().replace(0, np.nan)).clip(0, 3)

# 12. Cross-sectional rank spread (top-bottom momentum spread of each asset vs median)
mom20 = lnc - lnc.shift(20)
sig['xs_spread_mom20'] = mom20 - mom20.median(axis=1).to_frame().values

# 13. VIX change momentum (risk-off sensitivity of each asset via beta * dVIX)
vix_ret = vix.pct_change()
beta60 = ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var().replace(0, np.nan)
vix20 = vix / vix.shift(20) - 1.0
sig['vix_beta_20d'] = -beta60 * vix20

# 14. Semi-deviation ratio (downside deviation vs upside deviation)
for n in [20, 60]:
    pos = ret.clip(lower=0); neg = ret.clip(upper=0)
    up_dev = pos.rolling(n).std()
    dn_dev = (-neg).rolling(n).std()
    sig[f'semi_dev_ratio_{n}d'] = dn_dev / up_dev.replace(0, np.nan)

# 15. 120d momentum with vol condition (momentum only when vol low)
sig['mom120_lowvol_cond'] = (lnc - lnc.shift(120)) * (rv20 < rv20.rolling(120).median()).astype(float)

sig = {k: v for k, v in sig.items() if v is not None}
print(f"candidates: {len(sig)}")

results = {}
t0 = time.time()
for name, s in sig.items():
    s = s.reindex(close.index)
    res = {}
    for label, sl in [('full', slice(None)), ('recent_2y', close.index[-520:]), ('recent_1y', close.index[-260:])]:
        res[label] = eval_factor(s.loc[sl])
    results[name] = res
print(f"elapsed {time.time()-t0:.1f}s")

print(f"\n{'factor':26s} {'FULL ic1':>9s} {'icir1':>7s} {'hit':>5s} | {'2Y ic1':>7s} {'icir1':>7s} | {'1Y ic1':>7s} {'icir1':>7s} | {'cov':>5s}")
GATE_IC, GATE_ICIR = 0.0070, 0.0840
passed = []
for name, res in results.items():
    h1 = res['full'].get(1, {}); r2 = res['recent_2y'].get(1, {}); r1 = res['recent_1y'].get(1, {})
    line = (f"{name:26s} {h1.get('ic',np.nan):9.4f} {h1.get('icir',np.nan):7.3f} {h1.get('hit',np.nan):5.3f} | "
            f"{r2.get('ic',np.nan):7.4f} {r2.get('icir',np.nan):7.3f} | "
            f"{r1.get('ic',np.nan):7.4f} {r1.get('icir',np.nan):7.3f} | {res['full'].get('coverage',np.nan):5.3f}")
    print(line)
    ic, icir = h1.get('ic', np.nan), h1.get('icir', np.nan)
    if abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR and abs(r2.get('ic', np.nan)) >= GATE_IC:
        passed.append(name)

print("\nPASSED gates (|IC1|>=%.4f, |ICIR1|>=%.4f, 2Y |IC1|>=%.4f):" % (GATE_IC, GATE_ICIR, GATE_IC))
for p in passed:
    print("  ", p)

with open('scripts/miner3_scan_20320702.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner3_scan_20320702.json")
