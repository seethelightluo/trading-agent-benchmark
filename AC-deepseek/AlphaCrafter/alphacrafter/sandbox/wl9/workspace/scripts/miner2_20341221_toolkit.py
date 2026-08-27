"""Shared mining harness for miner_2: build 15-asset aligned panel and rank-IC validator."""
import pandas as pd, numpy as np, glob, os, json

ASSETS = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
VISIBLE = "2034-12-20"

def load_panel():
    panel = {}
    for a in ASSETS:
        fp = f"../persistent/stock_data/{a}.csv"
        df = pd.read_csv(fp)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        # only data up to visible date
        df = df[df.index <= pd.Timestamp(VISIBLE)]
        panel[a] = df
    return panel

def build_frame(panel):
    """Return wide DataFrame of close indexed by date with asset columns."""
    closes = {a: p['close'] for a, p in panel.items()}
    frame = pd.DataFrame(closes).sort_index()
    return frame

def compute_forward_returns(frame, horizon=10):
    """Forward return over `horizon` trading days per asset (aligned to asset's own rows)."""
    fwd = {}
    for a in frame.columns:
        s = frame[a]
        fwd[a] = s.shift(-horizon) / s - 1.0
    return pd.DataFrame(fwd)

def rank_ic(factor_df, fwd_df, min_valid=8):
    """Daily cross-sectional Spearman rank IC between factor and forward returns."""
    common_dates = factor_df.index.intersection(fwd_df.index)
    ics = []
    n_dates_ge8 = 0
    for dt in common_dates:
        f = factor_df.loc[dt]
        r = fwd_df.loc[dt]
        m = f.isna() | r.isna()
        fv, rv = f[~m], r[~m]
        if len(fv) >= min_valid:
            n_dates_ge8 += 1
            ic = fv.rank().corr(rv.rank())
            if not np.isnan(ic):
                ics.append(ic)
    ics = np.array(ics)
    ic_mean = ics.mean() if len(ics) else np.nan
    icir = (ic_mean / ics.std()) if len(ics) and ics.std() > 0 else np.nan
    hit = (ics > 0).mean() if len(ics) else np.nan
    return {"n_ic_dates": len(ics), "ic": ic_mean, "icir": icir,
            "ic_hit_ratio": hit, "n_dates_ge8": n_dates_ge8}

def run_validation(name, factor_df, horizon=10, min_valid=8):
    """factor_df: long DataFrame of values per date per asset column."""
    res = rank_ic(factor_df, forward_df, horizon)
    res['factor'] = name
    res['horizon'] = horizon
    return res

if __name__ == "__main__":
    panel = load_panel()
    print("Panel loaded for", len(panel), "assets, visible", VISIBLE)
    for a in panel:
        print(a, len(panel[a]))