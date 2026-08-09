# Extend the strict revalidation's mandatory novelty reconstruction with all subsequently
# admitted factor definitions available in the 2030-07-25 full-library repair script.
p='scripts/miner_2_20300808_revalidate_downside_close_location_participation_recovery_residual_20.py'
s=open(p,encoding='utf-8').read()
source=open('scripts/miner_2_20300725_tail_correlation_asymmetry_residual_60_full_library_repair.py',encoding='utf-8').read()
block=source.split('# Remaining admitted signals: reconstructed from their persisted definitions for complete 29-file screen.')[1].split("\nprint('FACTOR tail_correlation")[0]
# The four reconstructed entries are distinct from this revalidated candidate.
s=s.replace("\nprint('FACTOR downside_close_location", "\n# Full-library extension (definitions from prior validated repair)\n"+block+"\nprint('FACTOR downside_close_location")
open(p,'w',encoding='utf-8').write(s)
print('extended library reconstruction')
