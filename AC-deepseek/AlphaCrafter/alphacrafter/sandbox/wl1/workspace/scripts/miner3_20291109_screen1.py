"""miner3 2029-11-09: revalidate effective library + screen novel batch 3 (liquidity/tail/residual/macro-beta)."""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache_20291109.pkl')
C = panel['close']; O = panel['open']; H = panel['high']; L = panel['low']
R = panel['ret']; V = panel['vol']; M = panel['macro'].reindex(C.index).ffill()
gate_ic, gate_icir = 0.0070, 0.0840

def mom(nd, skip=1):
    return C.shift(skip) / C.shift(skip + nd) - 1.0

def vol(nd):
    return R.rolling(nd).std()

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

factors = {}
# ---------- existing effective library (re-validation) ----------
factors["mom_120d_skip5"] = mom(120, 5)
factors["vol_of_vol20x60"] = vol(20).rolling(60).std()
vix = M['VIX']; vix_ret = vix.pct_change()
beta_vix60 = R.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
factors["vix_beta_cond_60x20"] = -beta_vix60 * (vix / vix.shift(20) - 1.0).clip(-0.5, 0.5)
factors["nclv_1d"] = -(C.shift(1) / C - 1.0) * (L.shift(1) / C.shift(1) - 1.0).clip(-0.1, 0.1)
factors["rev_2d"] = -(C.shift(2) / C - 1.0)

# ---------- novel batch 3 ----------
ma20 = C.rolling(20).mean(); ma60 = C.rolling(60).mean(); ma200 = C.rolling(200).mean()
sd20 = R.rolling(20).std(); vol20 = vol(20); vol60 = vol(60)

# --- volume / liquidity family ---
vma20 = V.rolling(20).mean()
factors["vol_z_20"] = V / vma20 - 1.0
factors["vol_trend_5x20"] = V.rolling(5).mean() / vma20 - 1.0
factors["amihud_20"] = (R.abs() / (V + 1e-9)).rolling(20).mean()
factors["voladj_amihud_20"] = (R.abs() / (V + 1e-9)).rolling(20).mean() / vol20

# --- tail / risk asymmetry ---
factors["ret_consistency_20"] = (R > 0).astype(float).rolling(20).mean()
factors["sharpe_60"] = R.rolling(60).mean() / vol60
factors["skew_20"] = R.rolling(20).skew()
factors["tail5_60"] = R.rolling(60).quantile(0.05)
factors["updown_asym_20"] = (R.clip(lower=0).rolling(20).mean()) / (R.clip(upper=0).abs().rolling(20).mean() + 1e-9)
d = R.where(R < 0, 0.0)
factors["downside_ratio_60"] = np.sqrt((d ** 2).rolling(60).mean()) / vol60

# --- trend / mean-reversion variants ---
factors["dist_ma200"] = C / ma200 - 1.0
factors["bollinger_pos_20"] = (C - (ma20 - 2 * sd20)) / (4 * sd20 + 1e-9)
factors["bandwidth_20"] = (4 * sd20) / ma20
factors["dd_20d"] = C / C.rolling(20).max() - 1.0
factors["close_pos_range_5d"] = (C - L.rolling(5).min()) / (H.rolling(5).max() - L.rolling(5).min() + 1e-9)
factors["rev_1d_x_volz"] = -(C.shift(1) / C - 1.0) * (V.shift(1) / vma20.shift(1) - 1.0)
factors["rev_1d_x_calm"] = -(C.shift(1) / C - 1.0) * (vol(5) / vol20 - 1.0).clip(-0.5, 0.5)

# --- cross-sectional residual (idiosyncratic) ---
xmean_1d = R.mean(axis=1)
xmean_5d = R.rolling(5).mean().mean(axis=1)
factors["resid_ret_1d"] = -(R.shift(1) - xmean_1d.shift(1))
factors["resid_ret_5d"] = -(R.rolling(5).mean() - xmean_5d)

# --- cross-asset relative momentum ---
def_basket = mom(20, 1)[['XAU', 'US10Y', 'CN10Y']].mean(axis=1)
factors["rel_mom_def_20"] = mom(20, 1).sub(def_basket, axis=0)
factors["rel_mom_spx_20"] = mom(20, 1).sub(mom(20, 1)['SPX'], axis=0)

# --- macro-beta conditional ---
dxy = M['DXY']; usdcny = M['USDCNY']; usdjpy = M['USDJPY']
beta_dxy60 = R.rolling(60).cov(dxy.pct_change()) / dxy.pct_change().rolling(60).var()
beta_usdcny60 = R.rolling(60).cov(usdcny.pct_change()) / usdcny.pct_change().rolling(60).var()
beta_usdjpy60 = R.rolling(60).cov(usdjpy.pct_change()) / usdjpy.pct_change().rolling(60).var()
factors["dxy_beta_cond_60x20"] = -beta_dxy60 * (dxy / dxy.shift(20) - 1.0)
factors["usdcny_beta_cond_60x20"] = beta_usdcny60 * (usdcny / usdcny.shift(20) - 1.0)
factors["usdjpy_beta_cond_60x20"] = beta_usdjpy60 * (usdjpy / usdjpy.shift(20) - 1.0)
vix_ma60 = vix.rolling(60).mean()
factors["rev_1d_x_vixhi"] = -(C.shift(1) / C - 1.0) * (vix > vix_ma60).astype(float)

# --- correlation / beta regime ---
spx_ret = R['SPX']
factors["corr_spx_60"] = R.rolling(60).corr(spx_ret)
beta_spx60 = R.rolling(60).cov(spx_ret) / spx_ret.rolling(60).var()
factors["beta_spx60_chg_20"] = beta_spx60 - beta_spx60.shift(20)

# --- crypto rotation pair ---
btc_mom = mom(20, 1)['BTC']; eth_mom = mom(20, 1)['ETH']
factors["crypto_rel_rot_20d"] = (btc_mom - eth_mom).to_frame('BTC').join((eth_mom - btc_mom).to_frame('ETH')).reindex(columns=C.columns)

fwd = {h: C.shift(-h) / C - 1.0 for h in (1, 2, 3, 5)}
print(f"{'factor':28s} {'ic1':>7s} {'icir1':>7s} {'hit':>5s} {'n1':>5s} {'turn':>6s} | {'365d ic1':>9s} {'icir1':>7s} | {'120d ic1':>9s} {'icir1':>7s} | gate")
out = {}
for name, f in factors.items():
    entry = {'turn': turnover_ser(f)}
    for h in (1, 2, 3, 5):
        ics = ic_series_vec(f, fwd[h])
        entry[f'h{h}'] = stats(ics)
        if h == 1:
            ic1_ser = ics
    out[name] = entry
    if len(ic1_ser):
        last365 = ic1_ser[ic1_ser.index >= ic1_ser.index.max() - pd.Timedelta(days=365)]
        last120 = ic1_ser[ic1_ser.index >= ic1_ser.index.max() - pd.Timedelta(days=120)]
        ic365 = last365.mean() if len(last365) else np.nan
        icir365 = last365.mean() / last365.std() if len(last365) > 1 else np.nan
        ic120 = last120.mean() if len(last120) else np.nan
        icir120 = last120.mean() / last120.std() if len(last120) > 1 else np.nan
        s = entry['h1']
        passed = abs(s['ic']) >= gate_ic and abs(s['icir']) >= gate_icir
        print(f"{name:28s} {s['ic']:7.4f} {s['icir']:7.3f} {s['hit']:5.2f} {s['n']:5d} {entry['turn']:6.3f} | {ic365:9.4f} {icir365:7.3f} | {ic120:9.4f} {icir120:7.3f} | {'PASS' if passed else ''}")
    else:
        print(f"{name:28s} NO DATA")

with open('scripts/miner3_20291109_screen1_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("saved scripts/miner3_20291109_screen1_results.json")
