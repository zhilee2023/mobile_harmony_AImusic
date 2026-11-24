from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3


@dataclass(frozen=True)
class Spec:
    id: str
    start: int
    end: int
    traffic_bytes: int = 0
    scratch_bytes: int = 0
    flops: float = 0
    predicted_ms: float | None = None
    baseline: bool = False

    def validate(self, nodes):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', self.id):
            raise ValueError('Invalid candidate identifier')
        if not 0 <= self.start < self.end <= nodes:
            raise ValueError('Candidate must be a contiguous interval in the fixed serial order')
        if self.traffic_bytes < 0 or self.scratch_bytes < 0 or not math.isfinite(self.flops) or self.flops < 0:
            raise ValueError('Invalid logical traffic, scratch estimate or FLOPs')
        if self.predicted_ms is not None and (not math.isfinite(self.predicted_ms) or self.predicted_ms < 0):
            raise ValueError('Invalid predictor output')


@dataclass(frozen=True)
class Outcome:
    status: str
    latency_ms: float | None = None
    relative_l2: float | None = None
    source: str = 'unmeasured'

    @property
    def passed(self):
        return self.status == 'passed'

    def validate(self):
        if self.status not in {'passed', 'compile_failed', 'correctness_failed', 'device_failed'}:
            raise ValueError('A pending candidate is not a measured outcome')
        if self.passed and (self.latency_ms is None or not math.isfinite(self.latency_ms) or self.latency_ms < 0):
            raise ValueError('Passed outcome needs a finite nonnegative latency')
        if not self.passed and self.latency_ms is not None:
            raise ValueError('Failed outcomes cannot contribute timings to a plan')
        if self.relative_l2 is not None and (not math.isfinite(self.relative_l2) or self.relative_l2 < 0):
            raise ValueError('Numerical error must be finite and nonnegative')
        if self.source not in {'device_standalone', 'instrumented_callback', 'synthetic_test'}:
            raise ValueError('Measurement provenance must be explicit')


class Pool:
    """Metadata is public; candidate outcomes stay behind the query oracle.

    v1 deliberately models serial blocks with contiguous fusion intervals.
    It does not mislabel arbitrary-DAG fusion/peak liveness as implemented.
    """
    def __init__(self, nodes, specs, outcomes, signature, scope):
        if not isinstance(nodes, int) or not 1 <= nodes <= 256:
            raise ValueError('Serial block must have1–256 primitive nodes')
        if scope not in {'device_standalone', 'instrumented_callback_proxy', 'synthetic_test'}:
            raise ValueError('Unknown pool evidence scope')
        self.nodes, self.signature, self.scope = nodes, signature, scope
        self.specs = {s.id: s for s in specs}
        if len(self.specs) != len(specs) or not self.specs:
            raise ValueError('Candidate IDs must be nonempty and unique')
        self._outcomes = dict(outcomes)
        if set(self.specs) != set(self._outcomes):
            raise ValueError('Every precomputed candidate needs an explicit outcome')
        covered = set()
        for spec in specs:
            spec.validate(nodes)
            self._outcomes[spec.id].validate()
            if spec.baseline:
                if spec.end != spec.start+1 or not self._outcomes[spec.id].passed:
                    raise ValueError('Seed baseline must be a passed native singleton')
                covered.add(spec.start)
        if covered != set(range(nodes)):
            raise ValueError('Need a passed native singleton for each primitive')
        if not signature:
            raise ValueError('An input/device/precision signature is required')

    def seeds(self):
        return {s.id: self._outcomes[s.id] for s in self.specs.values() if s.baseline}

    def oracle(self):
        return QueryOracle(self)

    def full_outcomes_for_evaluation_only(self):
        return dict(self._outcomes)

    @classmethod
    def read(cls, path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if value.get('schemaVersion') != 1:
            raise ValueError('Unknown pool schema')
        return cls(value['nodes'], [Spec(**s) for s in value['candidates']],
                   {k: Outcome(**v) for k,v in value['outcomes'].items()}, value['signature'], value['scope'])


class QueryOracle:
    def __init__(self, pool):
        self._pool = pool
        self.calls = []

    def query(self, identifier):
        if identifier in self.calls or self._pool.specs[identifier].baseline:
            raise ValueError('A candidate may be revealed once; baseline is already known')
        self.calls.append(identifier)  # Failures consume the same query budget.
        return self._pool._outcomes[identifier]


class ObservationDatabase:
    """Append-only measurements partitioned by the full experiment signature."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS observations(signature TEXT,id TEXT,payload TEXT,PRIMARY KEY(signature,id))')

    @staticmethod
    def signature(metadata):
        required = {'device','modelSha256','mnnSha256','precision','threads','phase','inputSha256'}
        if not required <= metadata.keys():
            raise ValueError('Incomplete measurement signature')
        return hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def put(self, signature, identifier, payload):
        text = json.dumps(payload,sort_keys=True,allow_nan=False)
        with self.db:
            prior = self.db.execute('SELECT payload FROM observations WHERE signature=? AND id=?',(signature,identifier)).fetchone()
            if prior and prior[0] != text:
                raise ValueError('Do not overwrite a previous measurement; use a new trial ID')
            self.db.execute('INSERT OR IGNORE INTO observations VALUES(?,?,?)',(signature,identifier,text))

    def close(self):
        self.db.close()
