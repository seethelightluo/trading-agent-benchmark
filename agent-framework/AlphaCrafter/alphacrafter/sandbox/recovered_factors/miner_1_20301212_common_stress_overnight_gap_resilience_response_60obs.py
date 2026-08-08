"""Revalidation, single idea: common-stress overnight-gap resilience response."""
from pathlib import Path
src=Path('scripts/miner_1_20301128_common_stress_overnight_gap_resilience_response_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-11-27')", "END=pd.Timestamp('2030-12-11')")
src=src.replace("20301128_common_stress_overnight_gap_resilience_response", "20301212_common_stress_overnight_gap_resilience_response")
# Correct the historical harness's rank-turnover loop: position-independent labels
# and explicit finite filtering before pairwise rank correlation.
src=src.replace("rk=f.rank(axis=1,pct=True);to=[]\nfor i in range(1,len(rk)):\n q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()\n if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))\nprint(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}; concentration_mean_sd={f.std(axis=1).mean():.6f}')", "rk=f.rank(axis=1,pct=True);to=[]\nfor i in range(1,len(rk)):\n q=pd.DataFrame({'prior':rk.iloc[i-1].to_numpy(),'current':rk.iloc[i].to_numpy()}).replace([np.inf,-np.inf],np.nan).dropna()\n if len(q)>=8:\n  z=q['prior'].corr(q['current'],method='spearman')\n  if pd.notna(z): to.append(1-z)\nprint(f'turnover={np.mean(to):.6f}; turnover_pairs={len(to)}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}; concentration_mean_sd={f.std(axis=1).mean():.6f}')")
exec(compile(src,'gap_revalidation','exec'))
