"""Scheduled revalidation, one existing idea: residualized volatility-normalized drawdown repair speed."""
from pathlib import Path
src=Path('scripts/miner_1_20301017_residualized_volnorm_drawdown_repair_speed_20_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-10-02')", "END=pd.Timestamp('2030-12-25')")
src=src.replace("cutoff',END.date()", "REVALIDATION cutoff',END.date()")
src=src.replace("f.to_pickle('scripts/miner_1_20301017_residualized_volnorm_drawdown_repair_speed_20_60obs_candidate_signal.pkl')", "f.to_pickle('scripts/miner_1_20301226_residualized_volnorm_drawdown_repair_speed_20_60obs_revalidation_signal.pkl')")
# Correct only the non-essential turnover diagnostic so ranks retain aligned asset labels.
src=src.replace("q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()\n if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))", "q=pd.DataFrame({'prior':rk.iloc[i-1].to_numpy(),'current':rk.iloc[i].to_numpy()}).replace([np.inf,-np.inf],np.nan).dropna()\n if len(q)>=8:\n  c=q['prior'].corr(q['current'],method='spearman')\n  if pd.notna(c):to.append(1-c)")
exec(compile(src,'repair_speed_revalidation','exec'))
