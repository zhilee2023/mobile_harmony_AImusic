import statistics
import time
from .solver import solve,shortest_path,exact_pool_cg
from .search import discover


def study(pool,policies,budget,seeds,*,traffic_budget=None,scratch_budget=None,calibration=None):
    """Evaluate policies with a held-out outcome oracle after search completes."""
    tick=time.perf_counter()
    runs=[discover(pool,p,budget,seed=s,traffic_budget=traffic_budget,scratch_budget=scratch_budget,calibration=calibration) for p in policies for s in seeds]
    # The evaluation oracle is deliberately unavailable to policy selection.
    oracle=solve(pool,pool.full_outcomes_for_evaluation_only(),integer=True,traffic_budget=traffic_budget,scratch_budget=scratch_budget)
    certified=oracle.optimal and oracle.feasible and oracle.objective_ms is not None
    for run in runs:
        for row in run['trace']:
            row['fullPoolGapPercent']=None
            if certified and oracle.objective_ms>0 and row['bestModelledMs'] is not None:
                delta=row['bestModelledMs']-oracle.objective_ms
                if delta < -1e-6:raise ValueError('Search result cannot be better than a certified same-pool oracle')
                row['fullPoolGapPercent']=100*max(0,delta)/oracle.objective_ms
    summary={}
    for p in policies:
        values=[r['trace'][-1]['bestModelledMs'] for r in runs if r['policy']==p and r['trace'] and r['trace'][-1]['bestModelledMs'] is not None]
        summary[p]={'independentSeeds':len(seeds),'feasibleRuns':len(values),'meanModelledMs':statistics.mean(values) if values else None,'stdevModelledMs':statistics.stdev(values) if len(values)>1 else 0 if values else None}
    dp=None
    if traffic_budget is None:
        result=shortest_path(pool,pool.full_outcomes_for_evaluation_only(),scratch_budget=scratch_budget)
        dp={'objectiveMs':result.objective_ms,'selected':result.selected,'optimal':result.optimal}
    return {'schemaVersion':1,'scope':pool.scope,'poolSignature':pool.signature,'candidateCount':len(pool.specs),'nodeCount':pool.nodes,
            'restrictions':'Fixed serial order, contiguous interval kernels. Scratch is per-kernel working-set eligibility; traffic is a logical estimate budget. Not arbitrary DAG fusion, physical SRAM, measured DRAM or full-LLM latency.',
            'evaluationOracle':{'objectiveMs':oracle.objective_ms,'optimal':oracle.optimal,'selected':oracle.selected,'scope':'Finite provided pool, not all possible code implementations'},
            'dpBaseline':dp,'dpNotApplicableReason':'Aggregate traffic resource is enabled' if traffic_budget is not None else None,
            'exactKnownPoolCg':exact_pool_cg(pool,traffic_budget=traffic_budget,scratch_budget=scratch_budget),
            'runs':runs,'summary':summary,'elapsedSeconds':time.perf_counter()-tick}
