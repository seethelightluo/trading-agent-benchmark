"""miner3 2029-10-26: screen novel candidates batch 2 (macro aligned, more ideas)."""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache.pkl')
C = panel['close']; O = panel['open']; H = panel['high']; L = panel['low']
R = panel['ret']; V = panel['vol']; M = panel['macro']
gate_ic, gate_icir = 0.0070, 0.0840

# align macro to C index
M = M.reindex(C.index).ffill()

def mom(nd, skip=1):
    return C.shift(skip) / C.shift(skip + nd) - 1.0

def vol(nd):
    return R.rolling(nd).std()

factors = {}
# ---- macro-conditional (fixed alignment) ----
vix = M['VIX']; vix_ret = vix.pct_change()
beta_vix60 = R.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
factors["vix_beta_cond_60x20"] = -beta_vix60 * (vix / vix.shift(20) - 1.0).clip(-0.5, 0.5)
vix_ma60 = vix.rolling(60).mean()
vix_hi = (vix > vix_ma60).astype(float)
factors["mom20_x_vixhi"] = mom(20, 1) * vix_hi
factors["mom20_x_vixlo"] = mom(20, 1) * (1 - vix_hi)
dxy = M['DXY']; dxy_ret = dxy.pct_change()
beta_dxy60 = R.rolling(60).cov(dxy_ret) / dxy_ret.rolling(60).var()
factors["dxy_beta_60_x_dxyret20"] = beta_dxy60 * (dxy / dxy.shift(20) - 1.0)
us10y = C['US10Y']; d_us10y = us10y.diff()
beta_us10y60 = R.rolling(60).cov(d_us10y) / d_us10y.rolling(60).var()
factors["us10y_beta60_x_us10yret20"] = beta_us10y60 * (us10y / us10y.shift(20) - 1.0)

# ---- reversal / gap family ----
prev_close = C.shift(1)
overnight_gap = O / prev_close - 1.0
intraday = C / O - 1.0
factors["overnight_gap_1d"] = -overnight_gap            # gap reversion
factors["intraday_mom_1d"] = intraday                   # close vs open
factors["intraday_mom_5d"] = (C / O.shift(4) - 1.0)
factors["rev_1d_volspike"] = -(C.shift(1) / C - 1.0) * (vol(5) / vol(20) - 1.0)
factors["rev_3d_volspike"] = -(C.shift(3) / C - 1.0) * (vol(5) / vol(20) - 1.0)
# range-based reversal: prior day range position
prior_range_pos = (C.shift(1) - L.shift(1)) / (H.shift(1) - L.shift(1)).replace(0, np.nan)
factors["prev_range_pos_1d"] = -(prior_range_pos - 0.5)   # faded extremes
# volatility-scaled reversal
factors["rev_5d_voladj"] = -(C.shift(5) / C - 1.0) / vol(20)

# ---- trend quality / interaction ----
ma20 = C.rolling(20).mean(); ma60 = C.rolling(60).mean(); sd20 = R.rolling(20).std()
vol60 = vol(60)
factors["mom20_cond_lowvol"] = mom(20, 1) * (vol(20) < vol60).astype(float)
factors["mom60_cond_volspike"] = mom(60, 5) * (vol(10) / vol60 - 1.0).clip(-0.3, 0.3)
factors["zscore20_x_volspike"] = ((C - ma20) / sd20) * (vol(5) / vol(20) - 1.0)
# trend + stability composite (low vol-of-vol and positive trend)
vov = vol(20).rolling(60).std()
factors["trend_x_stability"] = mom(60, 5) * (-vov).rank(axis=1)
# rsi-like
delta = C.diff()
up = delta.clip(lower=0).rolling(14).mean()
dn = (-delta.clip(upper=0)).rolling(14).mean()
factors["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
# donchian position 20d
factors["donchian_pos20"] = (C - L.rolling(20).min()) / (H.rolling(20).max() - L.rolling(20).min())

# ---- crypto-pair tilt (BTC vs ETH) ----
btc_mom = mom(20, 1)['BTC']; eth_mom = mom(20, 1)['ETH']
crypto_gap = (btc_mom - eth_mom).to_frame('BTC').join((eth_mom - btc_mom).to_frame('ETH'))
factors["crypto_rel_gap_20d"] = crypto_gap.reindex(columns=C.columns)

def ic_series_vec(X, F):
    valid = X.notna() & F.notna()
    n = valid.sum(axis=1)
    keep = n >= 8
    Xr = X.rank(axis=1).where(valid)
    Fr = F.rank(axis=1).where(valid)
    dX = Xr.sub(Xr.mean(axis=1), axis=0)
    dF = Fr.sub(Fr.mean(axis=1), axis=0)
    num = (dX * dF).sum(axis=1)
    den = np.sqrt((dX ** 2).sum(axis=1) * (dF ** 2).sum(axis=1))
    ic = (num / den).where(den > 0)
    return ic[keep & ic.notna()]

def stats(ic_ser):
    if len(ic_ser) == 0:
        return dict(n=0, ic=float('nan'), icir=float('nan'), hit=float('nan'))
    return dict(n=len(ic_ser), ic=float(ic_ser.mean()),
                icir=float(ic_ser.mean() / ic_ser.std()) if len(ic_ser) > 1 else float('nan'),
                hit=float((ic_ser > 0).mean()))

def turnover_ser(F):
    s = F.rank(axis=1)
    chg = s.diff().abs().sum(axis=1) / s.notna().sum(axis=1)
    return float(chg.mean())

fwd1 = C.shift(-1) / C - 1.0
print(f"{'factor':28s} {'ic1':>7s} {'icir1':>7s} {'hit':>5s} {'n':>5s} {'turn':>6s} | {'365d ic':>8s} {'icir':>7s} | {'120d ic':>8s} {'icir':>7s} | gate")
out = {}
for name, f in factors.items():
    ic1 = ic_series_vec(f, fwd1)
    out[name] = {'h1': stats(ic1), 'turn': turnover_ser(f)}
    if len(ic1):
        last365 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=365)]
        last120 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=120)]
        ic365 = last365.mean() if len(last365) else np.nan
        icir365 = last365.mean() / last365.std() if len(last365) > 1 else np.nan
        ic120 = last120.mean() if len(last120) else np.nan
        icir120 = last120.mean() / last120.std() if len(last120) > 1 else np.nan
        passed = abs(out[name]['h1']['ic']) >= gate_ic and abs(out[name]['h1']['icir']) >= gate_icir
        print(f"{name:28s} {out[name]['h1']['ic']:7.4f} {out[name]['h1']['icir']:7.3f} {out[name]['h1']['hit']:5.2f} {out[name]['h1']['n']:5d} {out[name]['turn']:6.3f} | {ic365:8.4f} {icir365:7.3f} | {ic120:8.4f} {icir120:7.3f} | {'PASS' if passed else ''}")
    else:
        print(f"{name:28s} NO DATA")

with open('scripts/miner3_20291026_screen2_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("saved scripts/miner3_20291026_screen2_results.json")
