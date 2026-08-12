"""miner2 fast vectorized validation helpers (2029-03-16)."""
import pandas as pd, numpy as np

GATE_IC = 0.0070
GATE_ICIR = 0.0840
MIN_VALID = 8


def load_panel(path='scripts/panel_cache.pkl'):
    return pd.read_pickle(path)


def library_signals(panel):
    close = panel['close']
    high = panel['high']; low = panel['low']; open_ = panel['open']
    ret = panel['ret']
    lnc = np.log(close)
    sig = {}
    for nd in (1, 2, 3, 5):
        sig[f'rev_{nd}d'] = -(lnc - lnc.shift(nd))
        rng = high.rolling(nd).max() - low.rolling(nd).min()
        sig[f'nclv_{nd}d'] = -(close - low.rolling(nd).min()) / rng
    sig['nbody_1d'] = -(close - open_) / (high - low)
    sig['id_rev_1d'] = -(close / open_ - 1.0)
    rv20 = ret.rolling(20).std()
    sig['rev_1d_vs'] = -(lnc - lnc.shift(1)) / rv20
    sig['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
    sig['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
    macro = panel['macro']
    vix = macro['VIX']
    vix_ret = vix.pct_change()
    beta_vix = ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
    sig['vix_beta_cond_60x20'] = -beta_vix * (vix / vix.shift(20) - 1.0)
    return sig


def daily_ic_series(signal, fwd_ret, min_valid=MIN_VALID):
    """Vectorized daily Spearman IC: rank within each date, row-Pearson."""
    rs = signal.rank(axis=1, na_option='keep')
    rf = fwd_ret.rank(axis=1, na_option='keep')
    valid = signal.notna() & fwd_ret.notna()
    n = valid.sum(axis=1)
    rs_c = rs.sub(rs.where(valid).mean(axis=1), axis=0).where(valid)
    rf_c = rf.sub(rf.where(valid).mean(axis=1), axis=0).where(valid)
    num = (rs_c * rf_c).sum(axis=1)
    den = np.sqrt((rs_c ** 2).sum(axis=1) * (rf_c ** 2).sum(axis=1))
    ic = num / den
    ic[n < min_valid] = np.nan
    ic.name = 'ic'
    return ic


def ic_metrics(ic_series, signal, fwd_ret, horizons=(1, 2, 3, 5, 10, 20, 30), label=''):
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {'error': 'no valid dates'}
    out = {
        'n_dates': int(len(ic)),
        'ic': float(ic.mean()),
        'icir': float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0,
        'hit': float((ic > 0).mean()),
        'coverage': float(signal.notna().mean().mean()),
    }
    decay = {}
    for h in horizons:
        fwd = fwd_ret.shift(-h)
        ic_h = daily_ic_series(signal, fwd).dropna()
        if len(ic_h) > 0:
            decay[str(h)] = float(ic_h.mean())
    out['decay'] = decay
    yr = {}
    for y, grp in ic.groupby(ic.index.year):
        yr[str(y)] = {'ic': float(grp.mean()),
                      'icir': float(grp.mean() / grp.std()) if grp.std() > 0 else 0.0,
                      'n': int(len(grp))}
    out['by_year'] = yr
    for win, wname in [('365D', 'recent_12m'), ('182D', 'recent_6m')]:
        sub = ic[ic.index >= (ic.index.max() - pd.Timedelta(win))]
        if len(sub) >= 20:
            out[wname] = {'ic': float(sub.mean()),
                          'icir': float(sub.mean() / sub.std()) if sub.std() > 0 else 0.0,
                          'n': int(len(sub))}
    out['label'] = label
    return out


def signal_correlation_matrix(sig_new, lib_signals):
    rows = []
    new_stacked = sig_new.stack()
    for name, s in lib_signals.items():
        other = s.stack()
        both = pd.concat([new_stacked.rename('a'), other.rename('b')], axis=1).dropna()
        if len(both) < 30:
            rho = np.nan
        else:
            rho = float(both['a'].corr(both['b']))
        rows.append((name, rho))
    maxabs = max((abs(r) for _, r in rows if not np.isnan(r)), default=0.0)
    return maxabs, rows


def format_metrics(m, top=8):
    lines = [f"[{m.get('label','')}] n_dates={m.get('n_dates')} IC={m.get('ic'):.4f} ICIR={m.get('icir'):.3f} "
             f"hit={m.get('hit'):.3f} coverage={m.get('coverage'):.3f}"]
    dec = m.get('decay', {})
    if dec:
        lines.append("  decay(ic): " + " ".join(f"{k}d={v:.4f}" for k, v in sorted(dec.items(), key=lambda x: int(x[0]))))
    for w in ('recent_12m', 'recent_6m'):
        if w in m:
            lines.append(f"  {w}: IC={m[w]['ic']:.4f} ICIR={m[w]['icir']:.3f} n={m[w]['n']}")
    yr = m.get('by_year', {})
    if yr:
        ys = sorted(yr.items())
        lines.append("  by_year: " + " ".join(f"{y}:{v['ic']:.3f}/{v['icir']:.2f}" for y, v in ys))
    return "\n".join(lines)
