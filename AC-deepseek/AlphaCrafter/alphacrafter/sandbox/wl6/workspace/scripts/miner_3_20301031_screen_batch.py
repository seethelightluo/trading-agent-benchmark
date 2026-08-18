"""miner_3 2030-10-31: re-validate library + screen new factor families.
Data visible through 2030-10-30. Regime: VIX 37.7 (falling from 57.9 peak,
60d mean 43.1), SPX bull +11.95%/60d, extreme dispersion: ETH +53%/60d,
SOX bounce +5.3%/20d off -33.6%/60d, BTC +20.8%/60d but -7.0%/20d, China weak
(000300.SH -8.5%/60d). Frozen (HSI, 000688.SH, CN10Y) excluded from cross-
sectional stats.
"""
import json, time, base64, zlib, io
import numpy as np
import pandas as pd

VISIBLE = "2030-10-30"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8
IC_TH, ICIR_TH, CORR_TH = 0.0070, 0.0840, 0.5
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

t0 = time.time()


def load_close(sym, cutoff, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date").sort_index()


def load_panel(cutoff):
    closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
    for s in TRADABLE:
        df = load_close(s, cutoff)
        closes[s] = df["close"].astype(float)
        vols[s] = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
        highs[s] = df["high"].astype(float) if "high" in df else pd.Series(np.nan, index=df.index)
        lows[s] = df["low"].astype(float) if "low" in df else pd.Series(np.nan, index=df.index)
        opens[s] = df["open"].astype(float) if "open" in df else pd.Series(np.nan, index=df.index)
    return (pd.DataFrame(closes), pd.DataFrame(vols), pd.DataFrame(highs),
            pd.DataFrame(lows), pd.DataFrame(opens))


px, vol, hi, lo, op = load_panel(VISIBLE)
ret = px.pct_change()
obs = {s: load_close(s, VISIBLE, INDEX_DIR)["close"].astype(float) for s in OBS}

frozen = [s for s in TRADABLE if ret[s].dropna().iloc[-250:].abs().max() < 1e-12 or px[s].nunique() <= 1]
active = [s for s in TRADABLE if s not in frozen]
print("frozen assets:", frozen, "| active:", len(active), flush=True)


def rs(x, w):
    return x.rolling(w).std() * np.sqrt(252)


def beta_of(a, m, w):
    ra, rm_ = a.pct_change(), m.pct_change()
    return ra.rolling(w).cov(rm_) / rm_.rolling(w).var()


def corr_of(a, m, w):
    return a.pct_change().rolling(w).corr(m.pct_change())


# ---------------- library factor reconstruction (for correlation gate + re-validation) ----------------
lib_files = ['beta_chi_60d', 'beta_cn10y_60d', 'beta_vix_60d_neg', 'corr_us10y_60d',
             'down_vol_ratio_20x120', 'low_vol_20d', 'mom_10d_skip5', 'mom_120d_skip5',
             'sign_ewma_60d', 'skew_20d_neg', 'vix_beta_cond_60x20', 'vol_beta_spx_60d',
             'vol_of_vol20x60', 'vol_of_vol_chg_20d', 'xau_copper_cond_20d']


def load_lib_signal(fid):
    with open(f"factors/{fid}.json") as f:
        d = json.load(f)
    art = d.get("validation", {}).get("signal_artifact", {}) or d.get("signal_artifact", {})
    if not art or "data" not in art:
        return None
    raw = base64.b64decode(art["data"])
    txt = zlib.decompress(raw).decode()
    return pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True)


lib_signals = {}
for fid in lib_files:
    try:
        s = load_lib_signal(fid)
        if s is not None:
            lib_signals[fid] = s
    except Exception as e:
        print("lib load fail", fid, e, flush=True)
print("loaded library signals:", len(lib_signals), flush=True)


def max_lib_corr(fac):
    best = 0.0
    fv = fac.stack()
    for fid, ls in lib_signals.items():
        if fid in LIB_RECON and LIB_RECON[fid] is not None:
            ls = LIB_RECON[fid]
        lv = ls.stack().astype(float)
        j = pd.concat([fv, lv], axis=1, join="inner").dropna()
        if len(j) < 200:
            continue
        c = np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
        if np.isfinite(c):
            best = max(best, abs(c))
    return best


# ---------------- IC machinery ----------------
def fast_ic_series(factor, fwd, min_valid=MIN_INSTR):
    fr = px.pct_change(fwd).shift(-fwd)
    dates, ics = [], []
    for dt in factor.index:
        fv = factor.loc[dt]
        rv = fr.loc[dt]
        m = fv.notna() & rv.notna()
        m &= (ret.loc[dt].abs() > 1e-12)
        if m.sum() < min_valid:
            continue
        ic = np.corrcoef(fv[m], rv[m])[0, 1]
        if np.isfinite(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def ic_summary(ic):
    if len(ic) < 50:
        return np.nan, np.nan, np.nan, len(ic)
    m = ic.mean()
    sd = ic.std(ddof=1)
    return m, (m / sd if sd > 0 else np.nan), (ic > 0).mean(), len(ic)


def turnover_10d(f):
    fr = f.rank(axis=1, pct=True)
    return fr.diff().abs().mean(axis=1).mean()


def coverage_stats(f):
    valid = f.notna()
    return valid.stack().mean(), (valid.sum(axis=1) >= 8).mean()


# ---------------- library re-validation: reconstruct formula from JSON ----------------
def recon_lib_factor(fid):
    """Reconstruct factor panel from stored signal artifact when possible, else formula."""
    art = lib_signals.get(fid)
    if art is not None:
        return art.reindex(px.index).astype(float)
    return None


LIB_RECON = {}
for fid in lib_files:
    LIB_RECON[fid] = recon_lib_factor(fid)

print("\n=== LIBRARY RE-VALIDATION (h=10) ===", flush=True)
print(f"{'factor':<24}{'IC':>8}{'ICIR':>7}{'hit':>6}{'n':>6}  {'2027+':>16}  {'2029+':>16}  {'recent':>16}", flush=True)
lib_results = {}
sub_windows = {"2027+": "2027-01-01", "2029+": "2029-01-01", "recent": "2030-04-01"}
for fid in lib_files:
    f = LIB_RECON[fid]
    if f is None:
        print(f"{fid:<24} NO SIGNAL ARTIFACT (skip)", flush=True)
        continue
    ic = fast_ic_series(f, px.pct_change(H_ADMIT).shift(-H_ADMIT))
    m, ii, hit, n = ic_summary(ic)
    rec = {}
    for wname, wstart in sub_windows.items():
        icw = ic[ic.index >= wstart]
        mm, ii2, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii2, 4)) if nn > 50 else None
    lib_results[fid] = {"ic": m, "icir": ii, "hit": hit, "n": n, "sub": rec}
    s27 = rec.get("2027+", (None, None)); s29 = rec.get("2029+", (None, None)); srec = rec.get("recent", (None, None))
    print(f"{fid:<24}{m:>8.4f}{ii:>7.3f}{hit:>6.2f}{n:>6d}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>7.3f} "
          f"{s29[0] if s29 else float('nan'):>8.4f}{s29[1] if s29 else float('nan'):>7.3f} "
          f"{srec[0] if srec else float('nan'):>9.4f}{srec[1] if srec else float('nan'):>7.3f}", flush=True)

# ---------------- candidate factor families ----------------
F = {}
m5 = px.pct_change(5).shift(5)
m10 = px.pct_change(10).shift(5)
m20 = px.pct_change(20).shift(5)
m60 = px.pct_change(60).shift(5)

# --- G: relative strength / divergence ---
F["rel_mom_20d"] = m20.sub(m20.mean(axis=1), axis=0)
F["mom_accel_60x20"] = m60 - m20  # trend acceleration

# --- H: short-horizon reversal conditional on high VIX ---
vix = obs["VIX"]
vix_hi = (vix > vix.rolling(60, min_periods=30).median()).reindex(px.index).fillna(0.5).astype(float)
F["rev5_vix_hi"] = (-px.pct_change(5).shift(1)) * (0.5 + 0.5 * vix_hi)
F["rev10_vix_hi"] = (-px.pct_change(10).shift(3)) * (0.5 + 0.5 * vix_hi)

# --- I: vol term structure ---
F["vol_term_10x60"] = ret.rolling(10).std() / ret.rolling(60).std()
F["vol_skew_20d"] = ret.rolling(20).skew()

# --- J: rate sensitivities ---
F["beta_us10y_30d"] = beta_of(px, obs["US10Y"], 30)
F["corr_us10y_20d"] = corr_of(px, obs["US10Y"], 20)

# --- K: drawdown / crash distance ---
F["dist_high_60d"] = px / px.rolling(60, min_periods=30).max() - 1
F["maxdd_60d"] = (px / px.rolling(60, min_periods=30).max() - 1).rolling(60, min_periods=30).min()

# --- L: risk-adjusted momentum / Sharpe-style ---
F["risk_adj_mom_20d"] = m20 / rs(ret, 20)
F["risk_adj_mom_60d"] = m60 / rs(ret, 60)

# --- M: crypto rotation signals ---
F["crypto_rel_20d"] = (px["ETH"] / px["BTC"]).pct_change(20).shift(5).reindex(px.index)
F["crypto_rel_10d"] = (px["ETH"] / px["BTC"]).pct_change(10).shift(3).reindex(px.index)
F["eth_btc_div_mom"] = (px["ETH"].pct_change(20).shift(5) - px["BTC"].pct_change(20).shift(5)).reindex(px.index)

# --- N: trend persistence ---
ma20 = px.rolling(20, min_periods=10).mean()
F["above_ma20"] = (px > ma20).astype(float)
ma60 = px.rolling(60, min_periods=30).mean()
F["above_ma60"] = (px > ma60).astype(float)

# --- O: range / vol expansion ---
F["range_5d"] = (hi.rolling(5).max() - lo.rolling(5).min()) / px
F["range_20d"] = (hi.rolling(20).max() - lo.rolling(20).min()) / px

# --- P: NEW - SOX-style crash-rebound distance (drawdown recovery) ---
F["dd_recover_20d"] = px.pct_change(20).shift(5) - (px / px.rolling(60, min_periods=30).max() - 1)

# --- Q: NEW - vol-of-vol change (vol regime shift) ---
rv20 = ret.rolling(20).std()
F["vol_chg_20x60"] = rv20 / rv20.rolling(60).mean()

# --- R: NEW - VIX-beta conditional decay (short-horizon risk sensitivity) ---
vix_ret = vix.pct_change().reindex(px.index)
F["vix_beta_20d"] = ret.rolling(20).cov(vix_ret) / vix_ret.rolling(20).var()
F["vix_beta_neg_20d"] = -F["vix_beta_20d"]

# --- S: NEW - dispersion/relative strength vs WTI (energy complex) ---
F["rel_mom_wti_20d"] = m20.sub(px["WTI"].pct_change(20).shift(5), axis=0)

# --- T: NEW - 10d momentum of 20d momentum (momentum of momentum) ---
F["mom_of_mom_20x10"] = m20 - m20.shift(10)

fwd_all = {h: px.pct_change(h).shift(-h) for h in [1, 3, 5, 10, 20]}

results = {}
print(f"\n{'name':<24}{'IC':>8}{'ICIR':>7}{'hit':>6}{'n':>6}  {'librho':>7}{'turn':>7}  "
      f"{'2027+':>16}  {'recent':>16}  {'cov_ge8':>7}", flush=True)
sub_windows = {"2027+": "2027-01-01", "recent": "2030-04-01"}

for name, f in F.items():
    f = f[px.index]
    ic = fast_ic_series(f, fwd_all[H_ADMIT])
    m, ii, hit, n = ic_summary(ic)
    lc = max_lib_corr(f)
    turn = turnover_10d(f)
    cov_ad, cov_ge8 = coverage_stats(f)
    rec = {}
    for wname, wstart in sub_windows.items():
        icw = ic[ic.index >= wstart]
        mm, ii2, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii2, 4)) if nn > 50 else None
    dec = {}
    for h, fh in fwd_all.items():
        ich = fast_ic_series(f, fh)
        mm, ii2, _, _ = ic_summary(ich)
        dec[h] = (round(mm, 4), round(ii2, 4)) if np.isfinite(mm) else None
    results[name] = {"ic": m, "icir": ii, "hit": hit, "n": n, "librho": lc,
                     "turn": turn, "sub": rec, "decay": dec, "cov_ad": cov_ad, "cov_ge8": cov_ge8}
    s27 = rec.get("2027+", (None, None)); srec = rec.get("recent", (None, None))
    print(f"{name:<24}{m:>8.4f}{ii:>7.3f}{hit:>6.2f}{n:>6d}  {lc:>7.2f}  {turn:>7.3f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>7.3f} "
          f"{srec[0] if srec else float('nan'):>9.4f}{srec[1] if srec else float('nan'):>7.3f}  {cov_ge8:>7.2f}", flush=True)

print("\n--- candidates passing admission gate (|IC|>=%.4f, |ICIR|>=%.3f, n>=%d, cov_ge8>=0.5, librho<%.1f) ---"
      % (IC_TH, ICIR_TH, MIN_IC_DATES, CORR_TH), flush=True)
passed = []
for name, r in results.items():
    ok = (abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES
          and r["cov_ge8"] >= 0.5 and r["librho"] < CORR_TH)
    if ok:
        s27 = r["sub"].get("2027+", (0, 0)); srec = r["sub"].get("recent", (0, 0))
        stab = (s27[0] is not None and abs(s27[0]) >= IC_TH * 0.6) and (srec[0] is not None and abs(srec[0]) >= IC_TH * 0.6)
        passed.append((name, r, stab))
        print(f"  PASS {name:<24} ic={r['ic']:.4f} icir={r['icir']:.3f} librho={r['librho']:.3f} "
              f"turn={r['turn']:.3f} 2027+({s27[0]},{s27[1]}) recent({srec[0]},{srec[1]}) stab={stab}", flush=True)

with open("scripts/miner_3_20301031_screen_results.json", "w") as f:
    json.dump({"lib": lib_results, "candidates": results}, f, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)