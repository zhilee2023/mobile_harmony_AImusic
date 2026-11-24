from dataclasses import dataclass
import math
from .pool import Outcome


@dataclass
class Solution:
    objective_ms: float | None
    selected: list[str]
    feasible: bool
    optimal: bool
    duals: list[float] | None = None
    traffic_dual: float = 0
    artificial_mass: float = 0
    solver: str = 'CPLEX'


def admissible(pool, known, scratch_budget):
    return [s for s in pool.specs.values() if s.id in known and known[s.id].passed
            and (scratch_budget is None or s.scratch_bytes <= scratch_budget)]


def solve(pool, known, *, integer=False, traffic_budget=None, scratch_budget=None, seconds=30):
    """Exact-cover cost model, optional logical traffic budget, per-kernel scratch cap.

    Scratch is a max/eligibility filter, NOT an additive SRAM or process-RSS claim.
    LP Phase-I columns establish a basis but never count as a deployable plan.
    """
    try:
        import cplex
    except ImportError as error:
        raise RuntimeError('Install the optional IBM CPLEX Python package; no fabricated duals are used') from error
    specs = admissible(pool, known, scratch_budget)
    model = cplex.Cplex()
    for stream in ['log','results','warning','error']:
        getattr(model,'set_'+stream+'_stream')(None)
    model.parameters.threads.set(1)
    model.parameters.timelimit.set(seconds)
    if integer:
        model.parameters.mip.tolerances.mipgap.set(0)
        model.parameters.mip.tolerances.absmipgap.set(0)
    costs = [float(known[s.id].latency_ms) for s in specs]
    names = [s.id for s in specs]
    big_m = 1+sum(costs)
    artificial = [] if integer else ['__phase1_'+str(i) for i in range(pool.nodes)]
    names += artificial
    model.variables.add(names=names,obj=costs+[big_m]*len(artificial),lb=[0]*len(names),
                        ub=[1 if integer else cplex.infinity]*len(names),
                        types='B'*len(names) if integer else '')
    rows=[];rhs=[];senses=[]
    for node in range(pool.nodes):
        indices=[i for i,s in enumerate(specs) if s.start <= node < s.end]
        if artificial:indices.append(len(specs)+node)
        rows.append(cplex.SparsePair(ind=indices,val=[1.]*len(indices)));rhs.append(1);senses.append('E')
    if traffic_budget is not None:
        if traffic_budget < 0:raise ValueError('Negative traffic budget')
        rows.append(cplex.SparsePair(ind=list(range(len(specs))),val=[float(s.traffic_bytes) for s in specs]))
        rhs.append(float(traffic_budget));senses.append('L')
    model.linear_constraints.add(lin_expr=rows,senses=''.join(senses),rhs=rhs)
    model.solve()
    if not model.solution.is_primal_feasible():
        return Solution(None,[],False,False)
    values=model.solution.get_values();status=model.solution.get_status()
    optimal=status==101 if integer else status==1
    mass=sum(values[len(specs):])
    feasible=mass <= 1e-7
    selected=[s.id for i,s in enumerate(specs) if values[i] > 1e-7]
    objective=sum(values[i]*costs[i] for i in range(len(specs))) if feasible else None
    duals=None;traffic_dual=0.
    if not integer:
        if not optimal:raise RuntimeError('LP did not reach optimality; do not use uncertified prices')
        prices=model.solution.get_dual_values();duals=prices[:pool.nodes]
        if traffic_budget is not None:traffic_dual=prices[-1]
    return Solution(objective,selected,feasible,optimal,duals,traffic_dual,mass)


def shortest_path(pool, known, *, scratch_budget=None, traffic_budget=None):
    if traffic_budget is not None:
        raise ValueError('Plain DP baseline applies to the serial model without aggregate traffic constraint')
    distance=[math.inf]*(pool.nodes+1);plans=[[] for _ in distance];distance[0]=0
    specs=admissible(pool,known,scratch_budget)
    for start in range(pool.nodes):
        for spec in sorted((s for s in specs if s.start==start),key=lambda s:s.id):
            cost=distance[start]+known[spec.id].latency_ms
            if cost < distance[spec.end]:distance[spec.end]=cost;plans[spec.end]=plans[start]+[spec.id]
    ok=math.isfinite(distance[-1])
    return Solution(distance[-1] if ok else None,plans[-1],ok,ok,solver='serial-shortest-path')


def exact_pool_cg(pool, *, traffic_budget=None, scratch_budget=None):
    """Full measured cost library is already visible: finite-pool LP experiment.

    This is separate from budgeted discovery. Pricing sees actual costs here;
    those measurements are NOT called free online candidate discoveries.
    """
    all_outcomes=pool.full_outcomes_for_evaluation_only();known=pool.seeds();history=[]
    while True:
        lp=solve(pool,known,traffic_budget=traffic_budget,scratch_budget=scratch_budget)
        best=None
        for spec in admissible(pool,all_outcomes,scratch_budget):
            if spec.id in known:continue
            rc=all_outcomes[spec.id].latency_ms-sum(lp.duals[spec.start:spec.end])-lp.traffic_dual*spec.traffic_bytes
            if best is None or (rc,spec.id)<(best[0],best[1].id):best=(rc,spec)
        history.append({'knownColumns':len(known),'lpMs':lp.objective_ms,'artificialMass':lp.artificial_mass,
                        'minimumUnseenReducedCost':best[0] if best else None})
        if best is None or best[0] >= -1e-7:break
        known[best[1].id]=all_outcomes[best[1].id]
    full=solve(pool,all_outcomes,integer=True,traffic_budget=traffic_budget,scratch_budget=scratch_budget)
    restricted=solve(pool,known,integer=True,traffic_budget=traffic_budget,scratch_budget=scratch_budget)
    return {'scope':'Finite, already measured cost pool; modelled costs only', 'lpLowerBoundMs':lp.objective_ms if lp.feasible else None,
            'lpPricingComplete':True,'restrictedIntegerMs':restricted.objective_ms,'fullPoolIntegerOracleMs':full.objective_ms,
            'fullPoolOracleOptimal':full.optimal,'restrictedIntegerGloballyOptimalClaimed':False,'history':history}
