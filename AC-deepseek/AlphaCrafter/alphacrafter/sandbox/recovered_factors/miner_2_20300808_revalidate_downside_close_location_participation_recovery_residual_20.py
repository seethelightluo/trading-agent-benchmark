"""Revalidation of downside close-location participation recovery residual, single factor, as-of 2030-08-07."""
# The construction/validation implementation is deliberately identical to the admitted
# 2030-05-30 version; its cutoff is updated only to avoid changing factor definition.
p='scripts/miner_2_20300530_downside_close_location_participation_recovery_residual_20.py'
s=open(p,encoding='utf-8').read()
s=s.replace("E=pd.Timestamp('2030-05-29')", "E=pd.Timestamp('2030-08-07')")
s=s.replace('visible_through', 'REVALIDATION visible_through')
# Label full history periods clearly; no inference beyond cutoff (P is capped before fw returns).
open('scripts/miner_2_20300808_revalidate_downside_close_location_participation_recovery_residual_20.py','w',encoding='utf-8').write(s)
print('wrote revalidation script with strict completed-data cutoff 2030-08-07')
