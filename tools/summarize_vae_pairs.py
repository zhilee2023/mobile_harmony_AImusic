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


def summarize_factorial(campaign):
    cells = ('00', '10', '01', '11')
    runs = {cell: [run for run in campaign['runs'] if run['mode'] == cell] for cell in cells}
    if not campaign['pairs'] or any(len(sample) != len(campaign['pairs']) for sample in runs.values()):
        raise ValueError('Every block must contain all four rewrite combinations')
    if not all(run['screenAwake'] and run['pcmBitIdentical'] for run in campaign['runs']):
        raise ValueError('Every trial must pass screen and PCM checks')
    blocks = []
    for pair in campaign['pairs']:
        baseline, alias, epilogue, both = (pair['windowMs'][cell] for cell in cells)
        blocks.append(dict(aliasWithoutEpilogueMs=baseline - alias, aliasWithEpilogueMs=epilogue - both,
                           epilogueWithoutAliasMs=baseline - epilogue, epilogueWithAliasMs=alias - both,
                           interactionSavingMs=alias + epilogue - baseline - both))
    return dict(blocks=blocks,
                cells={cell: dict(medianMs=median(row['windowMs'] for row in sample),
                                  peakRssMiB=max(row['peakRssMiB'] for row in sample))
                       for cell, sample in runs.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--factorial', action='store_true')
    args = parser.parse_args()
    report = json.loads(args.campaign.read_bytes())
    result = summarize_factorial(report) if args.factorial else summarize(report)
    print(json.dumps(result, indent=2))
