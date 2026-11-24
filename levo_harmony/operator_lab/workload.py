"""Frequency-aware template costs, normalized BEFORE restricted-master pricing."""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class Exposure:
    calls_per_forward: int
    forwards: int

    @property
    def occurrences(self):
        if any(type(n) is not int or n<0 for n in [self.calls_per_forward,self.forwards]):
            raise ValueError('Exposure needs nonnegative integer counts')
        return self.calls_per_forward*self.forwards


@dataclass(frozen=True)
class Timing:
    ms: float
    unit: str = 'per_call'
    source: str = 'prediction'
    counted_occurrences: int | None = None


def aggregate_cost(exposures,timings,*,extra_overhead_ms=None):
    """Costs are MS PER TARGET WORKLOAD. Never multiply a preweighted total twice.

    None means overhead unmeasured, not zero. Caller can assert explicit zero for
    unchanged interfaces; this still does not turn predictions into measurements.
    """
    if not exposures or set(exposures)!=set(timings):raise ValueError('Every workload context needs a timing, including unchanged phases')
    terms={};known=0.;sources=set()
    for phase,exposure in exposures.items():
        t=timings[phase];n=exposure.occurrences
        if not math.isfinite(t.ms) or t.ms<0 or t.source not in {'prediction','standalone','graph_callback'}:raise ValueError('Invalid timing evidence')
        if t.unit=='per_call':
            if t.counted_occurrences is not None:raise ValueError('Per-call timing cannot already include occurrence weighting')
            value=n*t.ms
        elif t.unit=='workload_total':
            if t.counted_occurrences!=n:raise ValueError('Weighted total belongs to a different exposure')
            value=t.ms
        else:raise ValueError('Explicit per_call/workload_total timing unit required')
        terms[phase]=dict(callsPerForward=exposure.calls_per_forward,forwards=exposure.forwards,occurrences=n,costMs=value,source=t.source)
        known+=value;sources.add(t.source)
    if extra_overhead_ms is not None and (not math.isfinite(extra_overhead_ms) or extra_overhead_ms<0):raise ValueError('Overhead must be finite and nonnegative')
    return dict(costDomain='template_workload',contexts=terms,knownKernelCostMs=known,
                totalCostMs=None if extra_overhead_ms is None else known+extra_overhead_ms,
                extraOverheadMs=extra_overhead_ms,overheadsKnown=extra_overhead_ms is not None,
                evidence='graph_callback_proxy' if sources=={'graph_callback'} else 'prediction_or_standalone_proxy',
                wholeModelBenefitProven=False)


def family_exposure(baseline,profile,name_pattern,*,allow_extrapolation=False):
    """Normalize diagnostic repetitions, then expand to the target decode count."""
    b=baseline['native'];p=profile['native']
    if not baseline.get('screenAccepted') or not profile.get('screenAccepted'):raise ValueError('Workload/profile screen controls failed')
    if b['status']!='completed' or b['instrumented'] or p['status']!='completed' or not p['instrumented']:raise ValueError('Need separate baseline and instrumented profile')
    for field in ['caseSignature','precision','threads','mnnArchiveSha256']:
        if b[field]!=p[field]:raise ValueError('Workload/profile signature mismatch: '+field)
    rows=[r for r in p['operators']['nodes'] if re.fullmatch(name_pattern,r['name'])]
    observed={'prefill':len(p['rounds']),'decode':sum(r['decodeForwards'] for r in p['rounds'])}
    target={'prefill':1,'decode':b['requestedDecodeSteps']};exposures={};timings={};metadata={}
    if target['decode']>observed['decode'] and not allow_extrapolation:raise ValueError('Longer decode requires explicit latency/frequency extrapolation assumption; Attention needs KV buckets')
    for phase in target:
        group=[r for r in rows if r['phase']==phase];calls=sum(r['calls'] for r in group)
        if not group or calls<=0 or observed[phase]<=0 or calls%observed[phase]:raise ValueError('Cannot normalize profile call frequency')
        per_forward=calls//observed[phase];exposures[phase]=Exposure(per_forward,target[phase])
        timings[phase]=Timing(sum(r['seconds'] for r in group)*1000/calls,source='graph_callback')
        metadata[phase]=dict(observedCalls=calls,observedForwards=observed[phase],callSites=len({r['name'] for r in group}),extrapolated=target[phase]>observed[phase])
    return exposures,timings,metadata


def estimate_variant(baseline_cost,phase_ratios,*,unchanged_phases=(),extra_overhead_ms=None):
    if set(phase_ratios)!=set(baseline_cost['contexts']):raise ValueError('Missing phase-specific candidate estimate')
    contexts={};raw=0.
    for phase,term in baseline_cost['contexts'].items():
        ratio=phase_ratios[phase]
        if not math.isfinite(ratio) or ratio<0:raise ValueError('Invalid cost ratio')
        if phase in unchanged_phases:ratio=1.  # Ignore noisy standalone changes on an identical code path.
        value=term['costMs']*ratio;raw+=value;contexts[phase]=dict(**term,ratio=ratio,estimatedCostMs=value)
    if extra_overhead_ms is not None and (not math.isfinite(extra_overhead_ms) or extra_overhead_ms<0):raise ValueError('Invalid candidate overhead')
    base=baseline_cost['knownKernelCostMs'];known_saving=base-raw
    return dict(costDomain='template_workload',contexts=contexts,estimatedKernelCostMs=raw,
                savingBeforeUnmeasuredOverheadsMs=known_saving,
                estimatedNetSavingMs=None if extra_overhead_ms is None else known_saving-extra_overhead_ms,
                extraOverheadMs=extra_overhead_ms,eligibleForDeploymentFromEstimate=False,
                scope='Context callback baseline scaled by isolated ratios, heuristic only; not a measured whole-graph cost')
