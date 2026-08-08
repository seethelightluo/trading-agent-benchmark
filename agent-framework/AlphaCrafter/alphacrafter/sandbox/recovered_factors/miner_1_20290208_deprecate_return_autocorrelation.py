import json, os
src='factors/miner_1_20271104_return_autocorrelation_20obs.json'
dst=src+'_deprecated'
with open(src,encoding='utf-8') as h: x=json.load(h)
x['version']='2029-02-08 revalidation — DEPRECATED'
x['last_validated']='2029-02-08'
x['validation']['period']='2020-01-01 through 2029-02-07'
x['validation']['timestamp']='2029-02-08'
x['validation']['status']='DEPRECATED'
x['validation']['metrics'].update({'primary_horizon_days':10,'daily_paper_ic':0.022735769931092976,'daily_paper_icir':0.07337510621605826,'ic_hit_ratio':0.5115562403697997,'ic_dates':649,'universe_instruments':15,'minimum_instruments_per_ic_date':8,'mean_instruments_per_ic_date':12.043143297380585,'signal_cell_coverage':0.27686941356006106,'valid_asset_date_cells':12700,'mean_rank_turnover_10obs':0.580634749865519,'max_abs_library_correlation':0.11971990434161328,'max_correlation_factor':'downside_peer_correlation','decay':{'1d':{'ic':0.0003797236096358634,'icir':0.0012499658227346517,'dates':658},'5d':{'ic':0.0021498460254530246,'icir':0.007017501420199118,'dates':654},'10d':{'ic':0.022735769931092976,'icir':0.07337510621605826,'dates':649},'20d':{'ic':0.006991570610000324,'icir':0.0227959845687621,'dates':639}}})
x['validation']['regime_notes']='Revalidation failed the binding ICIR gate at the primary 10-session horizon (IC +0.02274, ICIR +0.07338; 649 IC dates, mean 12.04 instruments). It has also drifted adversely: latest 120 dates IC -0.08671 and ICIR -0.38147, after only modest 2028 evidence. Correlation evidence remains distinct (maximum reconstructed same-miner admitted-library Spearman |rho| 0.11972 versus downside_peer_correlation, 12,524 paired cells), but distinctness does not offset failed efficacy. Deprecated rather than retained.'
with open(dst,'w',encoding='utf-8') as h: json.dump(x,h,indent=2);h.write('\n')
os.remove(src)
print(dst)
