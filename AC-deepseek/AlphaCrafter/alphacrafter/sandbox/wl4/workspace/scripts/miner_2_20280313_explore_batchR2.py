"""miner_2 2028-03-13: revalidate 3 active factors + explore NEW candidate families.

Data visible through 2028-03-10 (per persistent/date.json).
Rank IC h=10 on the 15-asset cross-asset universe.
Gates (benchmark-wide): |IC|>=0.0070, |ICIR|>=0.0840; library correlation < 0.5.
"""
import json
import numpy as np
import pandas as pd

VISIBLE = pd.Timestamp('2028-03-10')
IC_TH, ICIR_TH = 0.0070, 0.0840
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def load_close():
    frames = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        frames[a] = df["close"]
    return pd.DataFrame(frames)


def load_ohlcv():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        out[a] = df
    return out


def load_macro():
    frames = {}
    for m in MACRO:
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        frames[m] = df["close"]
    return pd.DataFrame(frames)


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
        out.update(ic=np.nan, icir=np.nan, ic_std=np.nan, ic_hit=np.nan)
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
    return cov / var


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
    best, bestf = -1.0, None
    bestmean, bestmf = -1.0, None
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
panel = load_close()
ohlcv = load_ohlcv()
macro = load_macro()
rets = panel.pct_change()
mkt = rets.mean(axis=1)
vols = pd.DataFrame({a: ohlcv[a]["volume"] for a in ASSETS})

print(f"Data: {panel.shape[0]} days, {panel.shape[1]} assets, through {VISIBLE.date()}")

# ---------------- ACTIVE FACTORS ----------------
active = {}
down_mkt = mkt.where(mkt < 0, 0.0)
active["dn_mkt_beta_60d"] = rolling_beta(rets, down_mkt, 60, 40)
active["vol_price_corr_20"] = rets.rolling(20, min_periods=10).corr(vols)
cn10y_ret = panel["CN10Y"].pct_change()
active["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y_ret, 60, 40)

print("\n=== ACTIVE FACTOR REVALIDATION (visible through 2028-03-10) ===")
active_stats = {}
for fid, f in active.items():
    full = evaluate(f, panel, label=fid)
    r500 = evaluate(f, panel, label=fid + "_r500", valid_from=panel.index[-501])
    r250 = evaluate(f, panel, label=fid + "_r250", valid_from=panel.index[-251])
    active_stats[fid] = {"full": full, "r500": r500, "r250": r250}
    print(f"{fid}: full ic={full['ic']:+.4f} icir={full['icir']:+.4f} hit={full['ic_hit']:.3f} "
          f"n={full['n_ic_dates']} cov8={full['cov_ge8']:.3f} turn={full['turnover_10d']:.2f} | "
          f"r500 ic={r500['ic']:+.4f} icir={r500['icir']:+.4f} | r250 ic={r250['ic']:+.4f} icir={r250['icir']:+.4f}")

# ---------------- CANDIDATE EXPLORATION: batch R2 ----------------
print("\n=== CANDIDATE EXPLORATION: batch R2 (novel families) ===")
cand = {}
hl = pd.DataFrame({a: ohlcv[a]["high"] for a in ASSETS})
lo = pd.DataFrame({a: ohlcv[a]["low"] for a in ASSETS})
cl = panel

# R1: Price efficiency / trend quality (how much of the move is directional vs choppy)
for w in [10, 20, 40]:
    cum = np.log(cl / cl.shift(w))
    path = np.log(cl / cl.shift(1)).abs().rolling(w).sum()
    cand[f"efficiency_{w}"] = cum / path

# R2: High-low range position (where close sits inside recent range)
for w in [20, 60]:
    hh = hl.rolling(w).max()
    ll = lo.rolling(w).min()
    cand[f"hl_pos_{w}"] = (cl - ll) / (hh - ll).replace(0, np.nan)

# R3: Downside vs upside volatility asymmetry
for w in [20, 60]:
    dn = rets.where(rets < 0, 0.0).rolling(w).std()
    up = rets.where(rets > 0, 0.0).rolling(w).std()
    cand[f"vol_asym_{w}"] = dn / up.replace(0, np.nan)

# R4: Return skewness (crash risk)
for w in [20, 60]:
    cand[f"skew_{w}"] = rets.rolling(w, min_periods=10).skew()

# R5: Amihud-style price-impact (|ret|/volume), negated: low impact = more liquid = better
for w in [20, 60]:
    impact = (rets.abs() / vols.replace(0, np.nan)).rolling(w).mean()
    cand[f"amihud_{w}"] = -impact

# R6: Volume trend (short vs long volume)
vol_ma5 = vols.rolling(5).mean()
vol_ma20 = vols.rolling(20).mean()
vol_ma60 = vols.rolling(60).mean()
cand["volume_trend_5_20"] = vol_ma5 / vol_ma20
cand["volume_trend_20_60"] = vol_ma20 / vol_ma60

# R7: Cross-asset correlation to macro drivers (rolling corr, not beta)
dxy_ret = macro["DXY"].pct_change()
vix_ret = macro["VIX"].pct_change()
cand["corr_dxy_60"] = rets.rolling(60, min_periods=40).corr(dxy_ret)
cand["corr_vix_60"] = rets.rolling(60, min_periods=40).corr(vix_ret)
cand["corr_btc_60"] = rets.rolling(60, min_periods=40).corr(rets["BTC"])
cand["corr_spx_60"] = rets.rolling(60, min_periods=40).corr(rets["SPX"])
cand["corr_xau_60"] = rets.rolling(60, min_periods=40).corr(rets["XAU"])

# R8: Rate-change momentum (CN10Y 20/60d momentum, single-asset series broadcast cross-sectionally)
cand["cn10y_mom_20"] = cl["CN10Y"] / cl["CN10Y"].shift(20) - 1.0
cand["cn10y_mom_60"] = cl["CN10Y"] / cl["CN10Y"].shift(60) - 1.0

# R9: Drawdown depth (distance from 120d high)
cand["dd_120"] = cl / cl.rolling(120, min_periods=60).max() - 1.0

# R10: Parkinson volatility (intraday range-based)
pk = np.log(hl / lo).pow(2) / (4 * np.log(2))
cand["parkinson_vol_20"] = np.sqrt(pk.rolling(20).mean())
cand["parkinson_vol_60"] = np.sqrt(pk.rolling(60).mean())

# R11: Moving-average crossover strength
ema_f = cl.ewm(span=10).mean()
ema_s = cl.ewm(span=30).mean()
cand["ema_cross_10_30"] = (ema_f - ema_s) / cl

# R12: Range ratio (current range vs historical range) - compression/expansion
rng20 = (hl - lo).rolling(20).mean() / cl
rng60 = (hl - lo).rolling(60).mean() / cl
cand["range_ratio_20_60"] = rng20 / rng60

lib_panels = dict(active)

rows = []
for fid, f in cand.items():
    ev = evaluate(f, panel, label=fid)
    evr = evaluate(f, panel, label=fid + "_r250", valid_from=panel.index[-251])
    mc, mcf, mm, mmf = max_lib_corr(f, lib_panels)
    rows.append((fid, ev, evr, mc, mcf, mm, mmf))

for fid, ev, evr, mc, mcf, mm, mmf in rows:
    gate = "PASS" if (abs(ev["ic"]) >= IC_TH and abs(ev["icir"]) >= ICIR_TH) else "fail"
    corr_ok = "corr-OK" if (np.isfinite(mc) and mc < 0.5) else "CORR-HIGH"
    print(f"{fid:24s} ic={ev['ic']:+.4f} icir={ev['icir']:+.4f} hit={ev['ic_hit']:.3f} "
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
        print(f"{fid:24s} decay={dec}")

json.dump({"visible": str(VISIBLE.date()),
           "active": {fid: {k: (v if not isinstance(v, dict) else
                                {kk: vv for kk, vv in v.items() if kk not in ("label",)}) for k, v in s.items()}
                      for fid, s in active_stats.items()},
           "candidates": {fid: {"ic": ev["ic"], "icir": ev["icir"], "hit": ev["ic_hit"], "n": ev["n_ic_dates"],
                                "cov8": ev["cov_ge8"], "turn": ev["turnover_10d"],
                                "r250_ic": evr["ic"], "r250_icir": evr["icir"],
                                "maxcorr": mc, "maxcorr_factor": mcf, "meanabs": mm, "meanabs_factor": mmf}
                          for fid, ev, evr, mc, mcf, mm, mmf in rows},
           "decay": decay_store},
          open("scripts/_miner2_20280313_batchR2_results.json", "w"), indent=1, default=str)
print("\nsaved scripts/_miner2_20280313_batchR2_results.json")
