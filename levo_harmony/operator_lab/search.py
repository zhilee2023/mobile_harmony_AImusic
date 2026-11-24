import random
import math
import time
from .solver import solve

POLICIES={'random','greedy','flops','memory','roofline','dual','frozen_dual','dual_roofline'}


def roofline_ms(spec, calibration):
    if not calibration or not calibration.get('flops_per_second') or not calibration.get('bytes_per_second'):
        raise ValueError('Roofline needs explicit measured calibration; no default chip-peak numbers')
    if any(not math.isfinite(calibration[k]) or calibration[k]<=0 for k in ['flops_per_second','bytes_per_second']):
        raise ValueError('Roofline calibration must be finite and positive')
    return 1000*max(spec.flops/calibration['flops_per_second'],spec.traffic_bytes/calibration['bytes_per_second'])


def discover(pool, policy, budget, *, seed=42, traffic_budget=None, scratch_budget=None, calibration=None):
    """Hidden-library query study. Unknown measured costs never reach a selector.

    Dual pricing uses a predictor, therefore cannot certify a global LP bound.
    Query counts are distinct from precomputation compilations or agent tokens.
    """
    if policy not in POLICIES or budget < 0:raise ValueError('Invalid policy/budget')
    rng=random.Random(seed);oracle=pool.oracle();known=pool.seeds()
    frozen=solve(pool,known,traffic_budget=traffic_budget,scratch_budget=scratch_budget)
    baseline_node=[0.]*pool.nodes
    for spec in pool.specs.values():
        if spec.baseline:baseline_node[spec.start]=max(baseline_node[spec.start],known[spec.id].latency_ms)
    trace=[];start=time.perf_counter()
    for iteration in range(min(budget,len(pool.specs)-len(known))):
        lp=frozen if policy=='frozen_dual' else solve(pool,known,traffic_budget=traffic_budget,scratch_budget=scratch_budget)
        unseen=[s for s in pool.specs.values() if s.id not in known and (scratch_budget is None or s.scratch_bytes<=scratch_budget)]
        if not unseen:break
        def prediction(s):
            if policy in {'roofline','dual_roofline'}:return roofline_ms(s,calibration)
            return s.predicted_ms if s.predicted_ms is not None else .7*sum(baseline_node[s.start:s.end])
        def rank(s):
            if policy=='greedy':return (-sum(baseline_node[s.start:s.end]),s.id)
            if policy=='flops':return (-s.flops,s.id)
            if policy=='memory':return (-s.traffic_bytes,s.id)
            if policy=='roofline':return (prediction(s)-sum(baseline_node[s.start:s.end]),s.id)
            return (prediction(s)-sum(lp.duals[s.start:s.end])-lp.traffic_dual*s.traffic_bytes,s.id)
        candidate=rng.choice(sorted(unseen,key=lambda s:s.id)) if policy=='random' else min(unseen,key=rank)
        outcome=oracle.query(candidate.id);known[candidate.id]=outcome
        result=solve(pool,known,integer=True,traffic_budget=traffic_budget,scratch_budget=scratch_budget)
        trace.append({'queries':iteration+1,'candidate':candidate.id,'outcome':outcome.status,'bestModelledMs':result.objective_ms if result.feasible else None,
                      'selected':result.selected if result.feasible else [],'solverOptimal':result.optimal,'elapsedSeconds':time.perf_counter()-start})
    return {'policy':policy,'seed':seed,'poolScope':pool.scope,'queryCount':len(oracle.calls),'trace':trace,
            'compilationAttempts':None,'agentTokens':None,'globalLowerBoundMs':None,'globalOptimalityClaimed':False,
            'scope':'Finite-pool hidden-query simulation; costs of building the pool are separate. Best modelled plan is not measured whole-LLM latency.'}
