"""Calibrate IC/ICIR methodology against stored beta_vix_60d_neg artifact."""
import json, zlib, base64, io
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

VISIBLE = "2034-07-05"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

px = pd.DataFrame({s: pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"]).set_index("date")["close"].astype(float) for s in TRADABLE})
px = px[px.index <= pd.Timestamp(VISIBLE)].sort_index()
ret = px.pct_change()

d = json.load(open('factors/beta_vix_60d_neg.json'))
art = d['validation']['signal_artifact']
df = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(art['data'])).decode()), parse_dates=['date']).set_index('date')
print('artifact date range:', df.index.min().date(), '..', df.index.max().date(), 'rows:', len(df))
print('stored IC:', d['validation']['metrics']['ic'], 'ICIR:', d['validation']['metrics']['icir'], 'n:', d['validation']['metrics']['n_ic_dates'])

H = 10
fwd = px.shift(-H) / px - 1

def ic_series(fac, fwdmat, min_valid=8):
    dates, ics = [], []
    common = fac.index.intersection(fwdmat.index)
    for dt in common:
        f = fac.loc[dt]
        r = fwdmat.loc[dt]
        m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        # drop frozen (zero variance) assets
        if m.sum() < min_valid:
            continue
        fv = f[m].values.astype(float)
        rv = r[m].values.astype(float)
        if np.nanstd(fv) < 1e-12 or np.nanstd(rv) < 1e-12:
            continue
        rho, _ = spearmanr(fv, rv)
        if np.isfinite(rho):
            dates.append(dt)
            ics.append(rho)
    return pd.Series(ics, index=dates)

ics = ic_series(df, fwd)
print('recomputed n_ic_dates:', len(ics))
print('mean IC:', ics.mean())
print('std IC:', ics.std())
print('ICIR mean/std:', ics.mean()/ics.std())
print('ICIR mean/std*sqrt(n):', ics.mean()/ics.std()*np.sqrt(len(ics)))
print('hit ratio:', (ics>0).mean())
