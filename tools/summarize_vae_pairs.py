"""Summarize controlled VAE windows and bootstrap whole-window paired savings."""
import argparse
import json
import random
from pathlib import Path
from statistics import median, quantiles


def summarize(campaign, seed=42):
    paired = [pair['wholeWindowSavedMs'] for pair in campaign['pairs']]
    if len(paired) < 2 or len(campaign['runs']) != 2 * len(paired):
        raise ValueError('At least two complete pairs are required')
    if not all(run['screenAwake'] and run['pcmBitIdentical'] for run in campaign['runs']):
        raise ValueError('Every trial must pass screen and PCM checks')
    rng = random.Random(seed)
    resamples = [median(rng.choices(paired, k=len(paired))) for _ in range(10000)]
    bounds = quantiles(resamples, n=40, method='inclusive')
    metrics = {}
    for mode in ('baseline', 'fused'):
        times = [run['windowMs'] for run in campaign['runs'] if run['mode'] == mode]
        metrics[mode] = {'medianMs': median(times), 'observedP95Ms': quantiles(times, n=20, method='inclusive')[18],
                         'rtf': median(times) / (campaign['audioSeconds'] * 1000)}
    return dict(pairs=len(paired), windowByMode=metrics, pairedSavingsMs=paired, medianSavingMs=median(paired),
                exploratoryBootstrap95Ms=[bounds[0], bounds[-1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    print(json.dumps(summarize(json.loads(parser.parse_args().campaign.read_bytes())), indent=2))
