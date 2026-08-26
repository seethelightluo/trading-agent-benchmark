# fix third display
p='scripts/miner_2_20320726_contraction_reversal.py'
s=open(p).read();s=s.replace("[round(x['ic'].mean(),6) for x in np.array_split(o,3)]","[round(o.iloc[a:b].ic.mean(),6) for a,b in [(0,len(o)//3),(len(o)//3,2*len(o)//3),(2*len(o)//3,len(o))]]")
open(p,'w').write(s)
