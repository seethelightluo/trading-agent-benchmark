"""miner3 2029-07-20: explore novel candidate factors (screen batch), report IC/ICIR gates."""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache.pkl')
C = panel['close']; O = panel['open']; H = panel['high']; L = panel['low']
R = panel['ret']; V = panel['vol']
M = panel['macro']
gate_ic, gate_icir = 0.0070, 0.0840

def mom(nd, skip=1):
    return C.shift(skip) / C.shift(skip + nd) - 1.0

def vol(nd):
    return R.rolling(nd).std()

factors = {}
# --- trend/momentum family ---
factors["mom_20d_skip1"] = mom(20, 1)
factors["mom_60d_skip5"] = mom(60, 5)
factors["mom_120d_skip5"] = mom(120, 5)
# trend-conditioned momentum: momentum strength only counted in trend direction
ma60 = C.rolling(60).mean()
factors["mom_20d_cond_ma60"] = mom(20, 1) * np.sign(C - ma60)
factors["mom_60d_cond_ma60"] = mom(60, 5) * np.sign(C - ma60)
# vol-adjusted momentum (down-weights high-vol crypto)
factors["mom_20d_voladj"] = mom(20, 1) / vol(20)
factors["mom_60d_voladj"] = mom(60, 5) / vol(60)
# efficiency ratio (Kaufman): trend purity
factors["eff_ratio_20d"] = (C - C.shift(20)).abs() / R.abs().rolling(20).sum()
factors["eff_ratio_60d"] = (C - C.shift(60)).abs() / R.abs().rolling(60).sum()
# --- reversal / range family ---
factors["rev_1d"] = -(C.shift(1) / C - 1.0)
factors["rev_5d"] = -(C.shift(5) / C - 1.0)
factors["range_pos_20d"] = (C - L.rolling(20).min()) / (H.rolling(20).max() - L.rolling(20).min())
factors["dd_60d"] = C / C.rolling(60).max() - 1.0          # distance from 60d high
factors["dd_120d"] = C / C.rolling(120).max() - 1.0        # distance from 120d high
# --- vol / risk family ---
factors["vol_z_20x60"] = (vol(20) - vol(60)) / vol(60)
factors["vol_of_vol20x60"] = vol(20).rolling(60).std()
factors["maxdd_20d"] = R.rolling(20).min()
factors["skew_20d"] = R.rolling(20).skew()
factors["kurt_20d"] = R.rolling(20).kurt()
# --- volume family ---
factors["vol_trend_20x60"] = V.rolling(20).mean() / V.rolling(60).mean() - 1.0
# --- macro-beta family ---
us10y = C['US10Y']
d_us10y = us10y.diff()
cov60_us10y = R.rolling(60).cov(d_us10y)
var60_us10y = d_us10y.rolling(60).var()
factors["us10y_beta_60"] = cov60_us10y / var60_us10y
vix = M['VIX']; vix_ret = vix.pct_change()
cov60_vix = R.rolling(60).cov(vix_ret)
var60_vix = vix_ret.rolling(60).var()
factors["vix_beta_60"] = cov60_vix / var60_vix
# --- crypto relative strength (BTC vs ETH momentum gap, applied to both crypto names) ---
btc_mom = mom(20, 1)['BTC']; eth_mom = mom(20, 1)['ETH']
crypto_gap = (btc_mom - eth_mom).to_frame('BTC').join(
    (eth_mom - btc_mom).to_frame('ETH')).rename(columns={'BTC': 'BTC', 'ETH': 'ETH'})
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

print(f"{'factor':24s} {'h1 ic':>8s} {'icir':>7s} {'hit':>5s} {'n':>5s} | {'365d ic':>8s} {'icir':>7s} | {'120d ic':>8s} {'icir':>7s} | gate")
out = {}
for name, f in factors.items():
    row = {}
    for h in (1, 2, 3, 5):
        fwd = C.shift(-h) / C - 1.0
        row[h] = stats(ic_series_vec(f, fwd))
    ic1 = ic_series_vec(f, C.shift(-1) / C - 1.0)
    out[name] = row
    if len(ic1):
        last365 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=365)]
        last120 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=120)]
        ic365 = last365.mean() if len(last365) else np.nan
        icir365 = last365.mean() / last365.std() if len(last365) > 1 else np.nan
        ic120 = last120.mean() if len(last120) else np.nan
        icir120 = last120.mean() / last120.std() if len(last120) > 1 else np.nan
    else:
        ic365 = icir365 = ic120 = icir120 = np.nan
    ok = (abs(row[1]['ic']) >= gate_ic) and (abs(row[1]['icir']) >= gate_icir)
    print(f"{name:24s} {row[1]['ic']:+8.5f} {row[1]['icir']:+7.4f} {row[1]['hit']:5.3f} {row[1]['n']:5d} | "
          f"{ic365:+8.5f} {icir365:+7.4f} | {ic120:+8.5f} {icir120:+7.4f} | {ok}")

json.dump(out, open("scripts/miner3_20290720_screen_results.json", "w"), indent=1, default=str)
print("\nsaved scripts/miner3_20290720_screen_results.json")
