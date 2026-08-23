# corrected wrapper: the original exploratory calculation accidentally included post-current artifact rows;
# rerun identical logic through the supplied current date only.
exec(open('scripts/miner_1_20270218_vix_amplified_reversal.py').read().replace("prices=pd.DataFrame(px).sort_index();", "prices=pd.DataFrame(px).sort_index().loc[:pd.Timestamp('2027-02-18')];"))
