"""miner_2 shared harness (2035-10-11 cycle). 15-asset aligned panel + rank-IC validator + macro signals.
Visible window through prior completed trading day prior to current date.
Reads persistent csv (offline research; no live account)."""
import pandas as pd, numpy as np

ASSETS = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
VISIBLE = "2035-10-10"
STOCK = "../persistent/stock_data"
IDX = "../persistent/index_data"


def load_panel(visible=VISIBLE):
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"{STOCK}/{a}.csv", parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= pd.Timestamp(visible)]
        df = df.set_index("date").sort_index()
        out[a] = df
    return out


def build_frame(uni, col="close"):
    return pd.DataFrame({a: uni[a][col] for a in uni}).sort_index()


def load_macro(visible=VISIBLE, col_map=None):
    out = {}
    for name in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
        df = pd.read_csv(f"{IDX}/{name}.csv", parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= pd.Timestamp(visible)].set_index("date").sort_index()
        col = "close" if name == "VIX" else "pct_change"
        s = df[col]
        out["M_" + name] = s
    return out


def compute_forward_returns(frame, horizon=10):
    fwd = {}
    for a in frame.columns:
        s = frame[a]
        fwd[a] = s.shift(-horizon) / s - 1.0
    return pd.DataFrame(fwd)


def rank_ic(factor_df, fwd_df, min_valid=8):
    common = factor_df.index.intersection(fwd_df.index)
    ics = {}
    for dt in common:
        f, r = factor_df.loc[dt], fwd_df.loc[dt]
        m = f.isna() | r.isna()
        fv, rv = f[~m], r[~m]
        if len(fv) >= min_valid:
            ic = fv.rank().corr(rv.rank())
            if not np.isnan(ic):
                ics[dt] = ic
    s = pd.Series(ics)
    if len(s) == 0:
        return {"n_ic_dates": 0, "ic": np.nan, "icir": np.nan,
                "ic_hit_ratio": np.nan, "series": s}
    icm = s.mean(); icir = icm / s.std(ddof=1) if s.std(ddof=1) > 0 else np.nan
    return {"n_ic_dates": len(s), "ic": icm, "icir": icir,
            "ic_hit_ratio": (s > 0).mean(), "series": s}


def summarize(name, icres, window=520, start=None):
    s = icres.get('series')
    n = icres['n_ic_dates']
    return run_summary(name, s, icres, window, start)


def run_summary(name, s, icres=None, window=520, start=None):
    n = len(s) if s is not None else 0
    if n < 20:
        print(f"{name:28s}: TOO FEW ({n})")
        return None
    icm, icir, hit = s.mean(), s.mean() / s.std(ddof=1) if s.std(ddof=1) > 0 else 0, (s > 0).mean()
    extra = ""
    if window and len(s) >= window:
        r = s.iloc[-window:]
        ric = r.mean(); ricir = ric / r.std(ddof=1) if r.std(ddof=1) > 0 else 0
        extra = f"  recent{window}: ic={ric:+.4f} icir={ricir:+.4f} hit={(r>0).mean():.3f}"
    ok = abs(icm) >= 0.0070 and abs(icir) >= 0.084
    print(f"[{'OK' if ok else '--'}] {name:28s}: ic={icm:+.4f} icir={icir:+.4f} hit={hit:.3f} n={n:5d} [{s.index.min():%Y-%m-%d}~{s.index.max():%Y-%m-%d}]{extra}")
    return {"ic": float(icm), "icir": float(icir), "hit": float(hit), "n_dates": n,
            "first": str(s.index.min().date()), "last": str(s.index.max().date()),
            "series": s}


def rbeta(ret_df, macro, w):
    si = pd.concat([ret_df, macro.rename('M')], axis=1, join='inner')
    out = pd.DataFrame(index=si.index, columns=ret_df.columns)
    for a in si.columns:
        if a == 'M':
            continue
        cov = si[a].rolling(w, min_periods=max(30, w // 2)).cov(si['M'])
        var = si['M'].rolling(w, min_periods=max(30, w // 2)).var()
        out[a] = cov / var.replace(0, np.nan)
    return out


def rcorr(factor_s, macro_s, w):
    si = pd.concat([factor_s.rename('F'), macro_s.rename('M')], axis=1, join='inner')
    return si['F'].rolling(w, min_periods=max(30, w // 2)).corr(si['M'])