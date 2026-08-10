"""Screener: regime assessment + factor recent IC / correlation check (data through visible date only)."""
import json, glob, gzip, base64, io
import numpy as np
import pandas as pd

VISIBLE = "2026-07-15"
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

# ---------- prices ----------
px = {}
for a in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VISIBLE].set_index('date')['close']
    px[a] = df
px = pd.DataFrame(px)
rets = px.pct_change().dropna()

# ---------- market regime ----------
last = px.iloc[-1]
ma20 = px.tail(20).mean()
ma60 = px.tail(60).mean()
ma200 = px.tail(200).mean() if len(px) >= 200 else px.mean()
r20 = px.iloc[-1] / px.iloc[-21] - 1
r60 = px.iloc[-1] / px.iloc[-61] - 1
vol20 = rets.tail(20).std() * np.sqrt(252)
vol60 = rets.tail(60).std() * np.sqrt(252)
pos_vs_ma200 = last / ma200 - 1
ma20_slope = ma20 / ma20.shift(5).iloc[-1] - 1 if len(ma20) > 5 else np.nan

# cross-sectional dispersion (avg pairwise |corr|) on last 60d
c = rets.tail(60).corr()
avg_abs_corr = c.values[np.triu_indices(len(c), 1)].__abs__().mean()

# macro signals
macro = {}
for m in ['VIX','DXY','USDCNY','USDJPY','EURUSD']:
    try:
        df = pd.read_csv(f'../persistent/index_data/{m}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= VISIBLE]
        macro[m] = df
    except Exception as e:
        macro[m] = None

print("=" * 70)
print("MARKET REGIME (visible through", VISIBLE, ")")
print("=" * 70)
print(f"{'asset':<10} {'last':>10} {'r20%':>7} {'r60%':>7} {'vol20%':>7} {'vsMA200%':>8}")
for a in ASSETS:
    print(f"{a:<10} {last[a]:>10.1f} {r20[a]*100:>7.1f} {r60[a]*100:>7.1f} {vol20[a]*100:>7.1f} {pos_vs_ma200[a]*100:>8.1f}")

print("\nMedian r20: {:.2f}%  median r60: {:.2f}%  median vol20: {:.2f}%  median vsMA200: {:.2f}%".format(
    r20.median()*100, r60.median()*100, vol20.median()*100, pos_vs_ma200.median()*100))
print("Avg |pairwise corr| (60d): {:.3f}".format(avg_abs_corr))
print("Breadth: assets up over 20d: {}/15, up over 60d: {}/15".format((r20>0).sum(), (r60>0).sum()))
print("Equity-median (SPX/NDX/SOX/N225/HSI/CSI): 60d ret {:.2f}%".format(r60[['SPX','NDX','SOX','N225','HSI','000300.SH','000688.SH']].median()*100))

if macro.get('VIX') is not None:
    v = macro['VIX'].set_index('date')
    vcol = 'close' if 'close' in v.columns else v.columns[1]
    vix_last = v[vcol].iloc[-1]
    vix_prev = v[vcol].iloc[-21] if len(v) > 21 else v[vcol].iloc[0]
    print("\nVIX: last {:.1f} (20d ago {:.1f})  DXY last {:.2f}".format(
        vix_last, vix_prev, macro['DXY'].iloc[-1, 1] if macro.get('DXY') is not None else np.nan))

# ---------- factor library: decode signals, recent IC ----------
def decode_sa(path):
    d = json.load(open(path))
    sa = d['signal_artifact']
    raw = base64.b64decode(sa['data_b64'])
    mat = np.frombuffer(gzip.decompress(raw), dtype=np.float32).reshape(sa['n_dates'], sa['n_symbols'])
    dates = pd.date_range(sa['date_start'], sa['date_end'], periods=sa['n_dates'])
    sig = pd.DataFrame(mat, index=dates, columns=sa['symbols'])
    return d, sig

canon = [f for f in sorted(glob.glob('factors/*.json')) if not any(t in f for t in ['.bak', '.2026', '.2025'])]
print("\n" + "=" * 70)
print("FACTOR RECENT IC (rank IC vs next-day return, 2026-07-15 cutoff)")
print("=" * 70)
rows = []
sig_map = {}
for f in canon:
    try:
        d, sig = decode_sa(f)
    except Exception as e:
        print("skip", f, e); continue
    sig_map[d['factor_id']] = (d, sig)
    # align signals to price dates
    s = sig.reindex(px.index).iloc[:-1]          # signal at t
    r = rets.reindex(px.index).iloc[1:]          # return t+1
    common = s.index.intersection(r.index)
    ss, rr = s.loc[common], r.loc[common]
    ic_all = ss.apply(lambda col: col.corr(rr[col.name], method='spearman'))
    ic_recent = ss.tail(63).apply(lambda col: col.corr(rr.loc[col.index][col.name], method='spearman'))
    ic_30 = ss.tail(31).apply(lambda col: col.corr(rr.loc[col.index][col.name], method='spearman'))
    adm = d.get('benchmark_admission', {}).get('selected_metrics', {})
    q = adm.get('quality', 0)
    ic_sel = adm.get('ic', 0)
    icir_sel = adm.get('icir', 0)
    rows.append((d['factor_id'], d.get('factor_name',''), ic_all.mean(), ic_recent.mean(), ic_30.mean(), ic_sel, icir_sel, q, d.get('expected_direction', 1), d.get('tags', [])))
    print("{:<32} IC_all={:+.3f} IC_63d={:+.3f} IC_30d={:+.3f} | adm_ic={:+.3f} icir={:+.2f} q={:.4f} dir={}".format(
        d['factor_id'][:32], ic_all.mean(), ic_recent.mean(), ic_30.mean(), ic_sel, icir_sel, q, d.get('expected_direction', 1)))

# ---------- factor correlation (recent 63d) ----------
print("\n" + "=" * 70)
print("FACTOR SIGNAL PAIRWISE CORR (63d window, mean |corr|)")
print("=" * 70)
ids = [r[0] for r in rows]
corr_mat = pd.DataFrame(index=ids, columns=ids, dtype=float)
for i in ids:
    for j in ids:
        si = sig_map[i][1].reindex(px.index)
        sj = sig_map[j][1].reindex(px.index)
        cc = si.stack().to_frame('a').join(sj.stack().to_frame('b'))
        cc = cc.dropna()
        corr_mat.loc[i, j] = cc.tail(63*15).corr().iloc[0, 1] if len(cc) else np.nan
cvals = corr_mat.values[np.triu_indices(len(ids), 1)]
print("mean |pairwise corr|: {:.3f}  max: {:.3f}".format(np.nanmean(np.abs(cvals)), np.nanmax(np.abs(cvals))))
hi = np.argwhere(np.abs(corr_mat.values) > 0.7)
for i, j in hi:
    if i < j:
        print("  HI-CORR: {} <-> {} = {:.2f}".format(ids[i], ids[j], corr_mat.values[i, j]))
