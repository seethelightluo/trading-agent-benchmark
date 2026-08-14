"""Shared data loader for miner_1 research. Caps all data at the visible date
(current sim date minus one trading day) to avoid lookahead bias."""
import pandas as pd, numpy as np, json, os

UNIVERSE = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
            "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load_calendar():
    d = json.load(open('../persistent/date.json'))
    return d['current_date'], d['visible_through'], d['trading_days']

def load_panel(visible_through):
    """Return dict symbol -> DataFrame (date, open, high, low, close, volume) capped at visible_through."""
    out = {}
    for s in UNIVERSE:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= pd.Timestamp(visible_through)].copy()
        df = df.sort_values('date').reset_index(drop=True)
        out[s] = df
    return out

def load_macro(visible_through):
    out = {}
    for s in MACRO:
        df = pd.read_csv(f'../persistent/index_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= pd.Timestamp(visible_through)].copy()
        df = df.sort_values('date').reset_index(drop=True)
        out[s] = df
    return out

def build_close_matrix(panel):
    """Align closes on union of dates; returns (dates, DataFrame)."""
    closes = {}
    for s, df in panel.items():
        closes[s] = df.set_index('date')['close']
    mat = pd.concat(closes, axis=1).sort_index()
    mat = mat.dropna(how='all')
    return mat

def build_vol_matrix(panel):
    vols = {}
    for s, df in panel.items():
        vols[s] = df.set_index('date')['volume']
    mat = pd.concat(vols, axis=1).sort_index()
    mat = mat.dropna(how='all')
    return mat

def forward_returns(close_mat, h):
    """h-day forward simple returns aligned to factor date t -> return from t to t+h."""
    fwd = close_mat.shift(-h) / close_mat - 1.0
    return fwd

def rank_ic(factor_df, fwd_df):
    """Daily rank IC between factor values and forward returns across columns.
    Returns DataFrame of daily ICs (index=date), using dates with >=8 valid obs."""
    dates, ics = [], []
    for dt in factor_df.index:
        f = factor_df.loc[dt]
        r = fwd_df.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() >= 8:
            ic = np.corrcoef(f[m].rank(), r[m].rank())[0, 1]
            if np.isfinite(ic):
                dates.append(dt)
                ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def ic_summary(ic_series, label=""):
    ic = ic_series.dropna()
    n = len(ic)
    if n < 20:
        return f"{label}: n={n} too few"
    mean = ic.mean()
    std = ic.std(ddof=1) if n > 2 else np.nan
    icir = mean / std if std and std > 0 else np.nan
    hit = (ic > 0).mean()
    # recent half
    half = ic.iloc[n//2:]
    return (f"{label}: n={n} ic={mean:+.4f} icir={icir:+.3f} hit={hit:.3f} "
            f"recent_half_ic={half.mean():+.4f} recent_half_icir={half.mean()/half.std(ddof=1):+.3f}")

if __name__ == "__main__":
    cur, vis, tdays = load_calendar()
    print("current_date:", cur, "visible_through:", vis)
    panel = load_panel(vis)
    macro = load_macro(vis)
    cm = build_close_matrix(panel)
    print("close matrix shape:", cm.shape, "range:", cm.index.min().date(), "->", cm.index.max().date())
    # regime snapshot: last 20/60 trading days of the matrix
    last60 = cm.iloc[-60:]
    last20 = cm.iloc[-20:]
    r60 = (last60.iloc[-1] / last60.iloc[0] - 1).sort_values()
    r20 = (last20.iloc[-1] / last20.iloc[0] - 1).sort_values()
    print("\n60d returns:\n", r60.round(3).to_string())
    print("\n20d returns:\n", r20.round(3).to_string())
    vix = macro['VIX'].set_index('date')['close']
    vix_vis = vix[vix.index <= pd.Timestamp(vis)]
    print("\nVIX last 5:", vix_vis.tail(5).round(1).to_dict())
    print("VIX 60d ago:", round(float(vix_vis.iloc[-61]),1) if len(vix_vis)>61 else "na")
    print("breadth above MA20:", int((cm.iloc[-1] > cm.iloc[-21:-1].mean()).sum()), "/15")
    # flat-data artifact check on HSI/CN10Y recent
    for s in ["HSI","CN10Y"]:
        ser = cm[s].dropna()
        r = ser.pct_change().dropna()
        print(f"{s}: last 120d flat-frac={ (r.iloc[-120:].abs()<1e-12).mean():.3f} last close={ser.iloc[-1]:.2f}")
