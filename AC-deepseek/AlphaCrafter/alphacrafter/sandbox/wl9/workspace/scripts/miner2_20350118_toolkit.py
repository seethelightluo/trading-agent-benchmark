"""miner_2 shared harness (2035-01-18 cycle). Builds 15-asset aligned panel + rank-IC validator."""
import pandas as pd, numpy as np

ASSETS = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
VISIBLE = "2035-01-17"

def load_panel():
    panel = {}
    for a in ASSETS:
        fp = f"../persistent/stock_data/{a}.csv"
        df = pd.read_csv(fp)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[df.index <= pd.Timestamp(VISIBLE)]
        panel[a] = df
    return panel

def build_frame(panel):
    return pd.DataFrame({a: p['close'] for a, p in panel.items()}).sort_index()

def compute_forward_returns(frame, horizon=10):
    fwd = {}
    for a in frame.columns:
        s = frame[a]
        fwd[a] = s.shift(-horizon)/s - 1.0
    return pd.DataFrame(fwd)

def rank_ic(factor_df, fwd_df, min_valid=8):
    common = factor_df.index.intersection(fwd_df.index)
    ics, n_gte8 = [], 0
    for dt in common:
        f, r = factor_df.loc[dt], fwd_df.loc[dt]
        m = f.isna() | r.isna()
        fv, rv = f[~m], r[~m]
        if len(fv) >= min_valid:
            n_gte8 += 1
            ic = fv.rank().corr(rv.rank())
            if not np.isnan(ic):
                ics.append(ic)
    ics = np.array(ics)
    ic_m = ics.mean() if len(ics) else np.nan
    icir = (ic_m/ics.std()) if len(ics) and ics.std() > 0 else np.nan
    return {"n_ic_dates": len(ics), "ic": ic_m, "icir": icir,
            "ic_hit_ratio": (ics>0).mean() if len(ics) else np.nan, "n_dates_ge8": n_gte8}