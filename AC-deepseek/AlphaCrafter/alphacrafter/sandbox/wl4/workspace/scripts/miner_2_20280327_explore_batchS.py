"""miner_2 2028-03-27: revalidate 3 active factors + explore NEW candidate families (batch S).

Data visible through 2028-03-24 (API). Rank IC h=10 on the 15-asset cross-asset universe.
Gates (benchmark-wide): |IC|>=0.0070, |ICIR|>=0.0840; library correlation < 0.5
vs EFFECTIVE factors (vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d).
"""
import json
import glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

IC_TH, ICIR_TH = 0.0070, 0.0840
CORR_TH = 0.5
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def load_close():
    frames = {}
    for a in ASSETS:
        df = get_stock_daily_data(symbol=a, days=3000)
        if df is None or len(df) == 0:
            print(f"WARN no data {a}")
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        frames[a] = df.set_index("date")["close"].sort_index()
    return pd.DataFrame(frames).sort_index()


def load_ohlcv():
    out = {}
    for a in ASSETS:
        df = get_stock_daily_data(symbol=a, days=3000)
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        out[a] = df.set_index("date").sort_index()
    return out


def load_macro():
    frames = {}
    for m in MACRO:
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        frames[m] = df.set_index("date")["close"].sort_index()
    return pd.DataFrame(frames).sort_index()


def fwd_returns(panel, h=10):
    return panel.shift(-h) / panel - 1.0


def rank_ic_series(factor, fwd, min_valid=8):
    dates = factor.index.intersection(fwd.index)
    ics = {}
    for dt in dates:
        f = factor.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ic = f[mask].corr(r[mask], method="spearman")
        if np.isfinite(ic):
            ics[dt] = ic
    return pd.Series(ics, name="ic")


def evaluate(factor, panel, h=10, min_valid=8, label="f", valid_from=None, valid_to=None):
    fwd = fwd_returns(panel, h=h)
    ic = rank_ic_series(factor, fwd, min_valid=min_valid)
    if valid_from is not None:
        ic = ic[ic.index >= pd.Timestamp(valid_from)]
    if valid_to is not None:
        ic = ic[ic.index <= pd.Timestamp(valid_to)]
    out = {"label": label, "n_ic_dates": len(ic)}
    if len(ic) == 0:
        out.update(ic=np.nan, icir=np.nan, ic_std=np.nan, ic_hit=np.nan, cov_ge8=np.nan, turnover_10d=np.nan)
        return out
    icm = ic.mean()
    ics = ic.std(ddof=1)
    out.update(ic=float(icm), icir=float(icm / ics) if ics > 0 else np.nan,
               ic_std=float(ics), ic_hit=float((ic > 0).mean()))
    ge8 = (factor.notna().sum(axis=1) >= min_valid).mean()
    out["cov_ge8"] = float(ge8)
    ranks = factor.rank(axis=1)
    out["turnover_10d"] = float((ranks - ranks.shift(10)).abs().mean().mean())
    return out


def rolling_beta(ret, bench, win=60, minp=40):
    cov = ret.rolling(win, min_periods=minp).cov(bench)
    var = bench.rolling(win, min_periods=minp).var()
    return cov.div(var, axis=0)


def lib_corr_series(cand, lib, min_valid=8):
    common = cand.index.intersection(lib.index)
    rhos = []
    for dt in common:
        a = cand.loc[dt]
        b = lib.loc[dt]
        m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        if m.sum() < min_valid:
            continue
        r = a[m].corr(b[m], method="spearman")
        if np.isfinite(r):
            rhos.append(r)
    return pd.Series(rhos)


def max_lib_corr(cand, libs):
    best, bestf, bestmean, bestmf = -1.0, None, -1.0, None
    for name, lf in libs.items():
        s = lib_corr_series(cand, lf)
        if len(s) == 0:
            continue
        mx = s.abs().max()
        mn = s.abs().mean()
        if mx > best:
            best, bestf = mx, name
        if mn > bestmean:
            bestmean, bestmf = mn, name
    return best, bestf, bestmean, bestmf


# ---------------- load ----------------
print("loading...")
panel = load_close()
ohlcv = load_ohlcv()
macro = load_macro()
rets = panel.pct_change()
mkt = rets.mean(axis=1)
vols = pd.DataFrame({a: ohlcv[a]["volume"] for a in ASSETS})
hl = pd.DataFrame({a: ohlcv[a]["high"] for a in ASSETS})
lo = pd.DataFrame({a: ohlcv[a]["low"] for a in ASSETS})
op = pd.DataFrame({a: ohlcv[a]["open"] for a in ASSETS})
cl = panel

print(f"Data: {panel.shape[0]} days, {panel.shape[1]} assets, through {panel.index[-1].date()}")

# ---------------- ACTIVE FACTORS (recompute signals) ----------------
active = {}
down_mkt = mkt.where(mkt < 0, 0.0)
active["dn_mkt_beta_60d"] = rolling_beta(rets, down_mkt, 60, 40)
cn10y_ret = panel["CN10Y"].pct_change()
active["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y_ret, 60, 40)
mom20 = cl / cl.shift(20) - 1.0
mom60 = cl / cl.shift(60) - 1.0
vol20 = rets.rolling(20).std()
active["vol_adj_mom_accel_20x60"] = (mom20 - mom60) / vol20

print("\n=== ACTIVE FACTOR REVALIDATION (visible through 2028-03-24) ===")
active_stats = {}
for fid, f in active.items():
    full = evaluate(f, panel, label=fid)
    r500 = evaluate(f, panel, label=fid + "_r500", valid_from=panel.index[-501])
    r250 = evaluate(f, panel, label=fid + "_r250", valid_from=panel.index[-251])
    active_stats[fid] = {"full": full, "r500": r500, "r250": r250}
    print(f"{fid}: full ic={full['ic']:+.4f} icir={full['icir']:+.4f} hit={full['ic_hit']:.3f} "
          f"n={full['n_ic_dates']} cov8={full['cov_ge8']:.3f} turn={full['turnover_10d']:.2f} | "
          f"r500 ic={r500['ic']:+.4f} icir={r500['icir']:+.4f} | r250 ic={r250['ic']:+.4f} icir={r250['icir']:+.4f}")

lib_panels = dict(active)

# ---------------- CANDIDATE EXPLORATION: batch S ----------------
print("\n=== CANDIDATE EXPLORATION: batch S (novel families, h=10) ===")
cand = {}

# S1: Short-horizon reversal (negated returns at 1/3/5/10d)
cand["rev_1d"] = -rets
cand["rev_3d"] = -(cl / cl.shift(3) - 1.0)
cand["rev_5d"] = -(cl / cl.shift(5) - 1.0)
cand["rev_10d"] = -(cl / cl.shift(10) - 1.0)

# S2: Trend quality - fraction of up days (wins ratio) over 20/60d
cand["up_ratio_20"] = (rets > 0).rolling(20).mean()
cand["up_ratio_60"] = (rets > 0).rolling(60).mean()

# S3: Consecutive up/down streak (runs): sign of last return * streak length proxy
sign = np.sign(rets).fillna(0)
cand["streak_5"] = sign.rolling(5).sum()
cand["streak_10"] = sign.rolling(10).sum()

# S4: Return autocorrelation (trend persistence) over 20/60d
cand["autocorr_1_20"] = rets.rolling(20, min_periods=10).apply(
    lambda x: pd.Series(x).autocorr(1) if len(x) >= 10 else np.nan, raw=False)
cand["autocorr_1_60"] = rets.rolling(60, min_periods=20).apply(
    lambda x: pd.Series(x).autocorr(1) if len(x) >= 20 else np.nan, raw=False)

# S5: Vol-adjusted skewness (crash-risk adjusted)
sk20 = rets.rolling(20, min_periods=10).skew()
sk60 = rets.rolling(60, min_periods=20).skew()
cand["skew_vol_20"] = sk20 / vol20.replace(0, np.nan)
cand["skew_vol_60"] = sk60 / rets.rolling(60, min_periods=20).std().replace(0, np.nan)

# S6: Up/down capture ratio vs SPX (bull/bear asymmetry)
spx_ret = rets["SPX"]
upm = spx_ret.where(spx_ret > 0, 0.0)
dnm = spx_ret.where(spx_ret < 0, 0.0)
up_beta = rolling_beta(rets, upm, 60, 40)
dn_beta = rolling_beta(rets, dnm, 60, 40)
cand["updown_ratio_60"] = up_beta / dn_beta.replace(0, np.nan)

# S7: Beta to alternative benchmarks (BTC, XAU, US10Y, DXY, VIX)
btc_ret = rets["BTC"]
xau_ret = rets["XAU"]
us10y_ret = rets["US10Y"]
dxy_ret = macro["DXY"].pct_change().reindex(rets.index).ffill()
vix_ret = macro["VIX"].pct_change().reindex(rets.index).ffill()
cand["beta_btc_60"] = rolling_beta(rets, btc_ret, 60, 40)
cand["beta_xau_60"] = rolling_beta(rets, xau_ret, 60, 40)
cand["beta_us10y_60"] = rolling_beta(rets, us10y_ret, 60, 40)
cand["beta_dxy_60"] = rolling_beta(rets, dxy_ret, 60, 40)
cand["beta_vix_60"] = rolling_beta(rets, vix_ret, 60, 40)

# S8: Rate-curve beta: beta of returns to (US10Y - CN10Y) spread changes
spread = panel["US10Y"] - panel["CN10Y"]
cand["beta_spread_60"] = rolling_beta(rets, spread.pct_change(), 60, 40)

# S9: Drawdown recovery speed: time-since-high ratio (negative of days from high)
days_since_high = pd.DataFrame(index=cl.index, columns=cl.columns, dtype=float)
for a in ASSETS:
    s = cl[a]
    roll_max = s.cummax()
    # days since last new high (counting)
    dsh = np.nan
    counter = np.zeros(len(s))
    highs = s.values == roll_max.values
    # count days since last True
    idx = np.where(highs)[0]
    pos = np.searchsorted(idx, np.arange(len(s)))
    pos = np.clip(pos, 0, len(idx) - 1)
    days_ago = np.arange(len(s)) - idx[pos]
    days_ago[highs] = 0
    days_since_high[a] = days_ago
cand["days_since_high"] = -days_since_high

# S10: Range breakout strength: close relative to prior N-day high/low band
for w in [20, 60]:
    hh = hl.rolling(w).max().shift(1)
    ll = lo.rolling(w).min().shift(1)
    cand[f"breakout_{w}"] = (cl - ll) / (hh - ll).replace(0, np.nan)

# S11: Volume acceleration / trend (5d vs 20d volume ratio, 10d vs 60d)
vol_ma5 = vols.rolling(5).mean()
vol_ma20 = vols.rolling(20).mean()
vol_ma60 = vols.rolling(60).mean()
cand["vol_trend_5_20"] = vol_ma5 / vol_ma20
cand["vol_trend_10_60"] = vols.rolling(10).mean() / vol_ma60

# S12: Amihud illiquidity change (deterioration) - 20d vs 60d
imp20 = (rets.abs() / vols.replace(0, np.nan)).rolling(20).mean()
imp60 = (rets.abs() / vols.replace(0, np.nan)).rolling(60).mean()
cand["amihud_trend_20_60"] = imp20 / imp60

# S13: Intraday position: (close-open)/range - overnight vs intraday behavior
cand["intraday_pos_5"] = ((cl - op) / (hl - lo).replace(0, np.nan)).rolling(5).mean()

# S14: Overnight gap momentum: open/prev close - 1 (gap continuation)
gap = op / cl.shift(1) - 1.0
cand["gap_mom_10"] = gap.rolling(10).mean()

# S15: EWMA long-horizon momentum (40d half-life ~ ewm span 40)
cand["ewm_mom_40"] = cl.ewm(span=40).mean() / cl - 1.0
cand["ewm_mom_80"] = cl.ewm(span=80).mean() / cl - 1.0

# S16: Cross-asset dispersion regime (same value all assets -> skip for cross-sectional IC, but
# keep as conditional multiplier on other factors in strategy - record for info)

rows = []
for fid, f in cand.items():
    ev = evaluate(f, panel, label=fid)
    evr = evaluate(f, panel, label=fid + "_r250", valid_from=panel.index[-251])
    mc, mcf, mm, mmf = max_lib_corr(f, lib_panels)
    rows.append((fid, ev, evr, mc, mcf, mm, mmf))

for fid, ev, evr, mc, mcf, mm, mmf in rows:
    gate = "PASS" if (abs(ev["ic"]) >= IC_TH and abs(ev["icir"]) >= ICIR_TH) else "fail"
    corr_ok = "corr-OK" if (np.isfinite(mc) and mc < CORR_TH) else "CORR-HIGH"
    print(f"{fid:22s} ic={ev['ic']:+.4f} icir={ev['icir']:+.4f} hit={ev['ic_hit']:.3f} "
          f"n={ev['n_ic_dates']:5d} cov8={ev['cov_ge8']:.3f} turn={ev['turnover_10d']:.2f} | "
          f"r250 ic={evr['ic']:+.4f} icir={evr['icir']:+.4f} | "
          f"maxcorr={mc:.3f}({mcf}) meanabs={mm:.3f}({mmf}) => {gate}/{corr_ok}")

# decay for candidates passing IC/ICIR gate
print("\n=== DECAY (IC by horizon) for gate-passing candidates ===")
decay_store = {}
for fid, ev, evr, mc, mcf, mm, mmf in rows:
    if abs(ev["ic"]) >= IC_TH and abs(ev["icir"]) >= ICIR_TH:
        f = cand[fid]
        dec = {}
        for hh in [1, 2, 3, 5, 10, 20]:
            e = evaluate(f, panel, h=hh, label=fid)
            dec[str(hh)] = round(e["ic"], 4) if np.isfinite(e["ic"]) else None
        decay_store[fid] = dec
        print(f"{fid:22s} decay={dec}")

json.dump({"visible": str(panel.index[-1].date()),
           "active": {fid: {k: v for k, v in s.items() if k != "label"}
                      for fid, s in active_stats.items()},
           "candidates": {fid: {"ic": ev["ic"], "icir": ev["icir"], "hit": ev["ic_hit"], "n": ev["n_ic_dates"],
                                "cov_ge8": ev["cov_ge8"], "turnover_10d": ev["turnover_10d"],
                                "r250_ic": evr["ic"], "r250_icir": evr["icir"],
                                "max_corr": mc, "max_corr_factor": mcf, "mean_abs_corr": mm}
                          for fid, ev, evr, mc, mcf, mm, mmf in rows},
           "decay": decay_store},
          open("scripts/_miner2_batchS_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner2_batchS_results.json")
