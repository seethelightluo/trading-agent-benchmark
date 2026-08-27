"""miner_2 shared harness (2035-08-16 cycle). Builds 15-asset aligned panel + rank-IC validator + macro signals."""
import pandas as pd, numpy as np

ASSETS = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
VISIBLE = "2035-08-15"


def load_panel(visible=VISIBLE):
    from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
    WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
    uni = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=4000)
        if df is None or len(df) < 300:
            df = get_index_daily_data(symbol=s, days=4000)
        if df is not None and len(df) >= 300:
            df = df.copy(); df['date'] = pd.to_datetime(df['date']); df = df.set_index('date').sort_index()
            df = df[df.index <= pd.Timestamp(visible)]
            uni[s] = df
    return uni


def build_frame(uni):
    return pd.DataFrame({a: p['close'] for a, p in uni.items()}).sort_index()


def load_macro(visible=VISIBLE, fields=None):
    """Returns dict of macro pandas Series aligned+clipped."""
    spec = {'close': ['VIX'], 'pct_change': ['DXY','USDCNY','USDJPY','EURUSD']}
    out = {}
    for colname, files in spec.items():
        for fn in files:
            df = pd.read_csv(f'../persistent/index_data/{fn}.csv')
            df['date'] = pd.to_datetime(df['date'])
            s = df.set_index('date').sort_index()[colname]
            s = s[s.index <= pd.Timestamp(visible)]
            out['M_'+fn] = s
    return out


def compute_forward_returns(frame, horizon=10):
    fwd = {}
    for a in frame.columns:
        s = frame[a]
        fwd[a] = s.shift(-horizon)/s - 1.0
    return pd.DataFrame(fwd)


def rank_ic(factor_df, fwd_df, min_valid=8):
    """Returns info + per-date IC series (indexed by date)."""
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
    icm = s.mean(); icir = icm/s.std(ddof=1) if s.std(ddof=1) > 0 else np.nan
    return {"n_ic_dates": len(s), "ic": icm, "icir": icir,
            "ic_hit_ratio": (s > 0).mean(), "series": s}


def summarize(name, icres, window=520):
    s = icres.get('series')
    n = icres['n_ic_dates']
    if n < 20 or s is None or len(s) == 0:
        print(f"{name:28s}: TOO FEW ({n})")
        return None
    icm, icir, hit = icres['ic'], icres['icir'], icres['ic_hit_ratio']
    extra = ""
    if window and len(s) >= window:
        r = s.iloc[-window:]
        ric = r.mean(); ricir = ric/r.std(ddof=1) if r.std(ddof=1) > 0 else 0
        extra = f"  recent{window}: ic={ric:+.4f} icir={ricir:+.4f} hit={(r>0).mean():.3f}"
    ok = abs(icm) >= 0.0070 and abs(icir) >= 0.084
    print(f"[{'OK' if ok else '--'}] {name:28s}: ic={icm:+.4f} icir={icir:+.4f} hit={hit:.3f} n={n:5d} [{s.index.min():%Y-%m-%d}~{s.index.max():%Y-%m-%d}]{extra}")
    return icm, icir, hit


def rbeta(ret_df, macro, w):
    """expanding/rolling beta of each asset's return on macro return series."""
    si = pd.concat([ret_df, macro.rename('M')], axis=1, join='inner')
    out = pd.DataFrame(index=si.index, columns=ret_df.columns)
    for a in si.columns:
        if a == 'M':
            continue
        cov = si[a].rolling(w, min_periods=max(30, w//2)).cov(si['M'])
        var = si['M'].rolling(w, min_periods=max(30, w//2)).var()
        out[a] = cov/var.replace(0, np.nan)
    return out


def rcorr(factor_s, macro_s, w):
    si = pd.concat([factor_s.rename('F'), macro_s.rename('M')], axis=1, join='inner')
    return si['F'].rolling(w, min_periods=max(30, w//2)).corr(si['M'])