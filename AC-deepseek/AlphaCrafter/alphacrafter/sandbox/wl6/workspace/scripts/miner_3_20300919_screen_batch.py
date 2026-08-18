"""miner_3 2030-09-19: re-validate library + screen new factor families.
Data visible through 2030-09-18. Regime: VIX 47.3 elevated (off 57.9 peak), SPX
sideways at 5018 (+0.13%/60d), extreme dispersion: ETH +50%/60d, NDX +15%,
WTI +19.5%, XAU +6% vs SOX -26.5%, N225 -5.6%, BTC -5.5%. Frozen assets
(HSI, 000688.SH, CN10Y) excluded from cross-sectional stats.
"""
import json, time, base64, zlib, io
import numpy as np
import pandas as pd

VISIBLE = "2030-09-18"
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

# frozen = flat over last 250d
frozen = [s for s in TRADABLE if ret[s].dropna().iloc[-250:].abs().max() < 1e-12 or px[s].nunique() <= 1]
active = [s for s in TRADABLE if s not in frozen]
print("frozen assets:", frozen, "| active:", len(active), flush=True)

def rs(x, w):
    return x.rolling(w).std() * np.sqrt(252)

def beta_of(a, m, w):
    ra, rm_ = a.pct_change(), m.pct_change()
    cov = ra.rolling(w).cov(rm_)
    var = rm_.rolling(w).var()
    return cov / var

def corr_of(a, m, w):
    return a.pct_change().rolling(w).corr(m.pct_change())

# ---------------- library factor reconstruction (for correlation gate) ----------------
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
    df = pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True)
    return df

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
        lv = ls.stack()
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
    d = fr.diff().abs().mean(axis=1)
    return d.mean()

def coverage_stats(f):
    valid = f.notna()
    n_dates_ge8 = (valid.sum(axis=1) >= 8).mean()
    cov_ad = valid.stack().mean()
    return cov_ad, n_dates_ge8

# ---------------- factor definitions ----------------
F = {}

# --- Family G: high-dispersion relative strength / divergence ---
# cross-asset momentum dispersion: asset 20d mom minus cross-sectional mean (ranked)
m20 = px.pct_change(20).shift(5)
F["rel_mom_20d"] = m20.sub(m20.mean(axis=1), axis=0)
# winner-loser gap: current 60d momentum vs 20d momentum (trend acceleration)
m60 = px.pct_change(60).shift(5)
F["mom_accel_60x20"] = m60 - m20

# --- Family H: short-horizon reversal in high-vol regime (VIX-conditional) ---
vix = obs["VIX"]
vix_hi = (vix > vix.rolling(60, min_periods=30).median()).reindex(px.index).fillna(0.5).astype(float)
rev5 = -px.pct_change(5).shift(1)
F["rev5_vix_hi"] = rev5 * (0.5 + 0.5 * vix_hi)

# --- Family I: vol term structure / volatility of vol ---
F["vol_term_10x60"] = ret.rolling(10).std() / ret.rolling(60).std()
F["vol_skew_20d"] = ret.rolling(20).skew()

# --- Family J: rate / yield sensitivities ---
F["beta_us10y_30d"] = beta_of(px, obs["US10Y"], 30)
F["corr_us10y_20d"] = corr_of(px, obs["US10Y"], 20)

# --- Family K: drawdown / crash-distance ---
F["dist_high_60d"] = px / px.rolling(60, min_periods=30).max() - 1
F["maxdd_60d"] = (px / px.rolling(60, min_periods=30).max() - 1).rolling(60, min_periods=30).min()

# --- Family L: risk-adjusted momentum with vol penalty (Sharpe-style 20d) ---
F["risk_adj_mom_20d"] = px.pct_change(20).shift(5) / rs(ret, 20)

# --- Family M: ETH-BTC relative strength (crypto rotation) ---
F["crypto_rel_20d"] = (px["ETH"] / px["BTC"]).pct_change(20).shift(5).reindex(px.index)

# --- Family N: trend persistence: days since last close above/below 20d MA ---
ma20 = px.rolling(20, min_periods=10).mean()
F["above_ma20"] = (px > ma20).astype(float)

# --- Family O: 5d high-low range normalized (vol expansion) ---
F["range_5d"] = (hi.rolling(5).max() - lo.rolling(5).min()) / px

fwd_all = {h: px.pct_change(h).shift(-h) for h in [1, 3, 5, 10, 20]}

results = {}
print(f"\n{'name':<24}{'IC':>8}{'ICIR':>7}{'hit':>6}{'n':>6}  {'librho':>7}{'turn':>7}  "
      f"{'2027+':>16}  {'recent':>16}  {'cov_ge8':>7}", flush=True)
sub_windows = {"2027+": "2027-01-01", "recent": "2029-06-01"}

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
    print(f"{name:<24}{m:>8.4f}{ii:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.2f}  {turn:>7.3f}  "
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

with open("scripts/miner_3_20300919_screen_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
