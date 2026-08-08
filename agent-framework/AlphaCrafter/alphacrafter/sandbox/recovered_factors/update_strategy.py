p='strategy.py'
s=open(p).read()
s=s.replace('"""2031-07-24 admitted 10-factor bullish-recovery, medium-high-risk ensemble; completed bars only."""','"""2031-08-07 admitted 10-factor constructive-recovery, medium-high-risk ensemble; completed bars only."""')
s=s.replace('# defensive beta, WTI/Copper, signed-volume, lower partial moment, drawdown,\n# inverse VIX/EUR stress, residual autocorrelation, risk-adjusted trend, contrarian, oil shock.\nFW=(.14,.12,.11,.10,.10,.10,.10,.09,.07,.07)', '# defensive beta, lower partial moment, autocorrelation, VIX/EUR stress, WTI/Copper,\n# risk-adjusted trend, drawdown synchronization, residual jump concentration, contrarian, signed volume.\nFW=(.16,.13,.12,.11,.11,.10,.08,.07,.06,.06)')
s=s.replace('beta={}; signed={}; cross={}; lpm={}; draw={}; joint={}; auto={}; trend={}; contr={}; oil={}; vol={}', 'beta={}; signed={}; cross={}; lpm={}; draw={}; joint={}; auto={}; trend={}; contr={}; jump={}; vol={}')
old='''  if orr is not None:
   t=pd.concat([e,orr.rename("oil")],axis=1).dropna().tail(60); t["s"]=t.oil.clip(lower=0)
   def shockload(z,n):
    u=z.tail(n); q=float(u.s.var()); return float(u.iloc[:,0].cov(u.s)/q) if q>1e-14 else None
   x,y=shockload(t,20),shockload(t,60); oil[a]=y-x if x is not None and y is not None else None
'''
new='''  # Expansion of idiosyncratic jump concentration: largest residual moves share of energy.
  def jumpconc(n):
   u=e.tail(n).dropna()
   if len(u)<10: return None
   energy=u.pow(2); total=float(energy.sum())
   return float(energy.nlargest(max(1, len(energy)//5)).sum()/total) if total>1e-14 else None
  x,y=jumpconc(20),jumpconc(60); jump[a]=x-y if x is not None and y is not None else None
'''
assert old in s
s=s.replace(old,new)
s=s.replace('factors=(beta,cross,signed,lpm,draw,joint,auto,trend,contr,oil)', 'factors=(beta,lpm,auto,joint,cross,trend,draw,jump,contr,signed)')
open(p,'w').write(s)
