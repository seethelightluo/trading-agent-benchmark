"""miner_2 2027-12-20: revalidate 3 active factors + explore new candidates.

Data visible through 2027-12-17 (last completed trading day before 2027-12-20).
Rank IC h=10 on the 15-asset cross-asset universe.
Gates (benchmark-wide): |IC|>=0.0070, |ICIR|>=0.0840; library correlation < 0.5.
"""
import json
import numpy as np
import pandas as pd

VISIBLE = pd.Timestamp('2027-12-17')
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
        f = factor.loc[dt]; r = fwd.loc[dt]
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
    icm = ic.mean(); ics = ic.std(ddof=1)
    out.update(ic=float(icm), icir=float(icm/ics) if ics > 0 else np.nan,
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

def max_lib_corr(cand, lib_panels, min_valid=8):
    worst = {}
    for fid, lp in lib_panels.items():
        common = cand.index.intersection(lp.index)
        rhos = []
        for dt in common:
            a = cand.loc[dt]; b = lp.loc[dt]
            m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
            if m.sum() < min_valid:
                continue
            r = a[m].corr(b[m], method="spearman")
            if np.isfinite(r):
                rhos.append(r)
        if rhos:
            worst[fid] = float(np.max(np.abs(rhos)))
    if not worst:
        return 0.0, None
    fid = max(worst, key=worst.get)
    return worst[fid], fid

panel = load_close()
ohlcv = load_ohlcv()
macro = load_macro()
rets = panel.pct_change()
mkt = rets.mean(axis=1)

print(f"Data: {panel.shape[0]} days, {panel.shape[1]} assets, through {VISIBLE.date()}")

# ---------- ACTIVE FACTORS ----------
active = {}
# dn_mkt_beta_60d: beta(asset_ret, min(mkt_ret,0), 60)
down_mkt = mkt.where(mkt < 0, 0.0)
active["dn_mkt_beta_60d"] = rolling_beta(rets, down_mkt, 60, 40)
# vol_price_corr_20
vols = {a: ohlcv[a]["volume"] for a in ASSETS}
vol_df = pd.DataFrame(vols)
active["vol_price_corr_20"] = rets.rolling(20, min_periods=10).corr(vol_df)
# rate_beta_cn10y_60d
cn10y_ret = panel["CN10Y"].pct_change()
active["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y_ret, 60, 40)

print("\n=== ACTIVE FACTOR REVALIDATION ===")
for fid, f in active.items():
    full = evaluate(f, panel, label=fid)
    recent = evaluate(f, panel, label=fid+"_recent250", valid_from=panel.index[-251])
    recent500 = evaluate(f, panel, label=fid+"_recent500", valid_from=panel.index[-501])
    print(f"{fid}: full ic={full['ic']:.4f} icir={full['icir']:.4f} hit={full['ic_hit']:.3f} n={full['n_ic_dates']} cov8={full['cov_ge8']:.3f} turn={full['turnover_10d']:.2f}")
    print(f"   recent250: ic={recent['ic']:.4f} icir={recent['icir']:.4f} n={recent['n_ic_dates']}")
    print(f"   recent500: ic={recent500['ic']:.4f} icir={recent500['icir']:.4f} n={recent500['n_ic_dates']}")

# ---------- CANDIDATE BATCH ----------
print("\n=== CANDIDATE EXPLORATION ===")
cand = {}
# momentum family
cand["mom_20d_skip5"] = panel / panel.shift(25) - 1.0
cand["mom_60d_skip20"] = panel / panel.shift(80) - 1.0
cand["mom_120d_skip20"] = panel / panel.shift(140) - 1.0
cand["ema_10_30"] = panel.ewm(span=10).mean() / panel.ewm(span=30).mean() - 1.0
# beta family vs macro/asset drivers
btc_ret = rets["BTC"]; xau_ret = rets["XAU"]; wti_ret = rets["WTI"]
us10y_ret = panel["US10Y"].pct_change(); dxy_ret = macro["DXY"].pct_change()
vix_ret = macro["VIX"].pct_change(); usdjpy_ret = macro["USDJPY"].pct_change()
cand["btc_beta_60d"] = rolling_beta(rets, btc_ret, 60, 40)
cand["xau_beta_60d"] = rolling_beta(rets, xau_ret, 60, 40)
cand["wti_beta_60d"] = rolling_beta(rets, wti_ret, 60, 40)
cand["us10y_beta_60d"] = rolling_beta(rets, us10y_ret, 60, 40)
cand["dxy_beta_60d"] = rolling_beta(rets, dxy_ret, 60, 40)
cand["vix_beta_60d"] = rolling_beta(rets, vix_ret, 60, 40)
cand["usdjpy_beta_60d"] = rolling_beta(rets, usdjpy_ret, 60, 40)
cand["btc_corr_60"] = rets.rolling(60, min_periods=40).corr(btc_ret)
# price/vol structure
hl = pd.DataFrame({a: ohlcv[a]["high"] for a in ASSETS})
lo = pd.DataFrame({a: ohlcv[a]["low"] for a in ASSETS})
hh = hl.rolling(20, min_periods=10).max()
ll = lo.rolling(20, min_periods=10).min()
cand["hl_pos_20"] = (panel - ll) / (hh - ll)
cand["dd_60"] = panel / panel.rolling(60, min_periods=40).max() - 1.0
cand["dd_120"] = panel / panel.rolling(120, min_periods=60).max() - 1.0
cand["skew_20"] = rets.rolling(20, min_periods=10).skew()
# volatility family
vol20 = rets.rolling(20, min_periods=10).std()
vol60 = rets.rolling(60, min_periods=30).std()
cand["vol_ratio_20_60"] = vol20 / vol60
cand["inv_vol_20"] = -vol20  # low vol outperformance (sign: negative IC expected -> use -)
# realized range vol
cand["range_vol_20"] = (hl - lo) / panel

lib_panels = dict(active)  # 3 active factors as library

rows = []
for fid, f in cand.items():
    ev = evaluate(f, panel, label=fid)
    evr = evaluate(f, panel, label=fid+"_r250", valid_from=panel.index[-251])
    mc, mcf = max_lib_corr(f, lib_panels)
    rows.append((fid, ev, evr, mc, mcf))

for fid, ev, evr, mc, mcf in rows:
    gate = "PASS" if (abs(ev["ic"]) >= IC_TH and abs(ev["icir"]) >= ICIR_TH) else "fail"
    corr_ok = "corr-OK" if (np.isfinite(mc) and mc < 0.5) else "CORR-HIGH"
    print(f"{fid:22s} ic={ev['ic']:+.4f} icir={ev['icir']:+.4f} hit={ev['ic_hit']:.3f} n={ev['n_ic_dates']:5d} cov8={ev['cov_ge8']:.3f} turn={ev['turnover_10d']:.2f} | r250 ic={evr['ic']:+.4f} icir={evr['icir']:+.4f} | maxcorr={mc:.3f}({mcf}) => {gate}/{corr_ok}")

# decay for best candidates
print("\n=== DECAY (IC by horizon) for PASS candidates ===")
for fid, ev, evr, mc, mcf in rows:
    if abs(ev["ic"]) >= IC_TH and abs(ev["icir"]) >= ICIR_TH and np.isfinite(mc) and mc < 0.5:
        f = cand[fid]
        dec = {}
        for hh in [1, 2, 3, 5, 10, 20]:
            e = evaluate(f, panel, h=hh, label=fid)
            dec[str(hh)] = round(e["ic"], 4) if np.isfinite(e["ic"]) else None
        print(f"{fid:22s} decay={dec}")

json.dump({fid: {"ic": ev["ic"], "icir": ev["icir"], "hit": ev["ic_hit"], "n": ev["n_ic_dates"],
                 "cov8": ev["cov_ge8"], "turn": ev["turnover_10d"],
                 "r250_ic": evr["ic"], "r250_icir": evr["icir"],
                 "maxcorr": mc, "maxcorr_factor": mcf}
           for fid, ev, evr, mc, mcf in rows},
          open("scripts/_miner2_20271220_results.json", "w"), indent=1)
print("\nsaved scripts/_miner2_20271220_results.json")
