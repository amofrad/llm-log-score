"""Exploratory efficiency comparison; never modifies manuscript artifacts.

Run with the Project Python environment. See threshold_efficiency_theory.md
for guarantee scopes and interpretation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta, binom

import threshold_split_sensitivity as prior

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/exploratory/threshold_efficiency'
DELTA = .05
ALPHAS = (.20, .25, .30, .35, .40)
DETERMINISTIC = ('bonferroni', 'holm', 'fallback', 'multistart')
TIED = ('tied_bonferroni', 'tied_fallback')
GAMMAS = (1., .75, .50, .25)


def upper_bound(k, n, tail):
    k, n = np.asarray(k), np.asarray(n)
    if np.any(k != np.floor(k)) or np.any(n != np.floor(n)):
        raise ValueError('Exact binomial bounds require integer counts')
    if np.any((k < 0) | (k > n)) or not 0 < tail < 1:
        raise ValueError('Invalid binomial counts or tail')
    out = np.ones(n.shape)
    good = (n > 0) & (k < n)
    out[good] = beta.ppf(1-tail, k[good]+1, n[good]-k[good])
    return out


def candidate_family(grid, tied=False):
    if tied:
        return np.repeat(grid, len(GAMMAS)), np.tile(GAMMAS, len(grid))
    return np.asarray(grid), np.ones(len(grid))


def acceptance(u, thresholds, gammas, coins=None):
    u = np.asarray(u)
    if coins is None:
        assert np.all(gammas == 1)
        return u[:, None] >= thresholds[None, :]
    return ((u[:, None] > thresholds[None, :]) |
            ((u[:, None] == thresholds[None, :]) &
             (coins[:, None] <= gammas[None, :])))


def counts(mask, z, indices):
    selected = mask[indices]
    return selected.sum(axis=0), ((1-z[indices, None])*selected).sum(axis=0)


def reject(pvals, method, thresholds):
    """Candidates are ordered from least to most restrictive."""
    m = len(pvals)
    rejected = np.zeros(m, dtype=bool)
    levels = np.full(m, DELTA/m)
    method = method.removeprefix('tied_')
    if method == 'bonferroni':
        rejected = pvals <= levels
    elif method == 'holm':
        for j, i in enumerate(np.argsort(pvals, kind='stable')):
            levels[i] = DELTA/(m-j)
            if pvals[i] > levels[i]:
                break
            rejected[i] = True
    elif method == 'fallback':
        for i in range(m-1, -1, -1):
            if pvals[i] <= levels[i]:
                rejected[i] = True
                if i:
                    levels[i-1] += levels[i]
    elif method == 'multistart':
        starts = [int(np.flatnonzero(np.isclose(thresholds, j/10))[0])
                  for j in range(1, 11)]
        levels[:] = DELTA/len(starts)
        for start in starts:
            for i in range(start, -1, -1):
                if pvals[i] > levels[i]:
                    break
                rejected[i] = True
    else:
        raise ValueError(method)
    return rejected, levels


def select(n, k, alpha, method, thresholds):
    pvals = np.where(n > 0, binom.cdf(k, n, alpha), 1.)
    rejected, levels = reject(pvals, method, thresholds)
    indices = np.flatnonzero(rejected & (n > 0))
    return (int(indices[0]) if len(indices) else None), pvals, levels, rejected


def test_metrics(n, k, n_test, index):
    if index is None:
        nn = kk = 0
    else:
        nn, kk = int(n[index]), int(k[index])
    return {'n_test': n_test, 'n_answered': nn, 'n_incorrect': kk,
            'answer_rate': nn/n_test,
            'observed_error': kk/nn if nn else np.nan,
            'error_lo': (0. if kk == 0 else beta.ppf(.025, kk, nn-kk+1)) if nn else np.nan,
            'error_hi': (1. if kk == nn else beta.ppf(.975, kk+1, nn-kk)) if nn else np.nan}


def expected_test(u, z, threshold, gamma):
    if threshold is None:
        return {'expected_answer_rate': 0., 'weighted_error_ratio': np.nan}
    weights = (u > threshold).astype(float) + gamma*(u == threshold)
    return {'expected_answer_rate': float(weights.mean()),
            'weighted_error_ratio': float(weights @ (1-z)/weights.sum()) if weights.sum() else np.nan}


def summaries(frame):
    repeated = frame[frame.seed != prior.SEED]
    keys = ['grader', 'model_key', 'allocation', 'method', 'coin_rep', 'alpha']
    output = []
    for key, g in repeated.groupby(keys, sort=False):
        errors = g.observed_error.dropna()
        output.append(dict(zip(keys, key)) | {
            'n_partitions': len(g), 'n_supported': int(g.supported.sum()),
            'mean_answer_rate': g.answer_rate.mean(),
            'mean_expected_answer_rate': g.expected_answer_rate.mean(),
            'median_error': errors.median(), 'error_q25': errors.quantile(.25),
            'error_q75': errors.quantile(.75), 'n_answering': len(errors),
            'n_test_errors_above_alpha': int((errors > key[-1]).sum()),
            'mean_common_answer_rate': g.common_answer_rate.mean()})
    return pd.DataFrame(output)


def paired_summary(frame, baseline):
    keys = ['grader', 'model_key', 'allocation', 'seed', 'alpha']
    cols = keys+['supported', 'answer_rate', 'observed_error', 'expected_answer_rate']
    merged = frame.merge(baseline[cols], on=keys, suffixes=('', '_baseline'), validate='many_to_one')
    rows = []
    groupkeys = ['grader', 'model_key', 'allocation', 'method', 'coin_rep', 'alpha']
    for key, g in merged[merged.seed != prior.SEED].groupby(groupkeys, sort=False):
        delta = g.answer_rate-g.answer_rate_baseline
        expected_delta = g.expected_answer_rate-g.expected_answer_rate_baseline
        both = g[g.observed_error.notna() & g.observed_error_baseline.notna()]
        rows.append(dict(zip(groupkeys, key)) | {
            'n_partitions': len(g), 'mean_answer_rate_gain': delta.mean(),
            'mean_expected_answer_rate_gain': expected_delta.mean(),
            'gain_q25': delta.quantile(.25), 'gain_q75': delta.quantile(.75),
            'n_more_answers': int((delta > 1e-12).sum()),
            'n_fewer_answers': int((delta < -1e-12).sum()),
            'n_support_gained': int((g.supported & ~g.supported_baseline).sum()),
            'n_support_lost': int((~g.supported & g.supported_baseline).sum()),
            'n_both_answering': len(both),
            'mean_error_change_when_both_answer': (both.observed_error-both.observed_error_baseline).mean()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['deterministic', 'ties'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, sources, original_refs, bound_refs = {}, [], {}, {}
    for grader in prior.GRADERS:
        frames, grid, refs, bounds, source = prior.load_grader(grader)
        inputs[grader] = frames; sources.append(source)
        original_refs[grader] = refs; bound_refs[grader] = bounds
    saved = pd.read_csv(ROOT/'results/exploratory/threshold_split_sensitivity/all_results.csv')
    references = saved.set_index(['grader', 'model_key', 'allocation', 'seed', 'alpha'])
    tied = args.stage == 'ties'
    thresholds, gammas = candidate_family(grid, tied)
    methods = TIED if tied else DETERMINISTIC
    rows, primary_bounds = [], []
    checked_bounds = checked_results = 0
    for partition_index, seed in enumerate(range(prior.SEED, prior.SEED+101)):
        for coin_rep in (range(5) if tied else [-1]):
            coins = (np.random.default_rng(np.random.SeedSequence([20260914, seed, coin_rep])).random(4326)
                     if tied else None)
            for grader, frames in inputs.items():
                for model, (ids, u, z) in frames.items():
                    mask = acceptance(u, thresholds, gammas, coins)
                    assert np.all(mask[:, :-1] >= mask[:, 1:]), 'Rule family is not nested'
                    assert not mask[u == -1].any()
                    if tied:
                        assert np.array_equal(mask[:, ::len(GAMMAS)], acceptance(u, grid, np.ones(len(grid))))
                    for allocation, tenths in prior.ALLOCATIONS.items():
                        sel, test, common = prior.split_indices(len(ids), seed, tenths)
                        n, k = counts(mask, z, sel)
                        nt, kt = counts(mask, z, test)
                        nc, kc = counts(mask, z, common)
                        upper = upper_bound(k, n, DELTA/len(thresholds))
                        info = {'grader': grader, 'model_key': model, 'allocation': allocation,
                                'seed': seed, 'coin_rep': coin_rep, 'n_selection': len(sel),
                                'family_size': len(thresholds)}
                        if not tied and allocation == '50-50':
                            for i, t in enumerate(grid):
                                reference = bound_refs[grader][(model, seed, float(t))]
                                assert int(reference['n_cert_answered']) == n[i]
                                assert int(reference['k_cert_incorrect']) == k[i]
                                assert abs(float(reference['risk_upper'])-upper[i]) < 1e-11
                                checked_bounds += 1
                        if seed == prior.SEED:
                            for i, (t, gamma) in enumerate(zip(thresholds, gammas)):
                                primary_bounds.append(info | {'threshold':t, 'gamma':gamma,
                                    'n_answered':int(n[i]), 'n_incorrect':int(k[i]), 'bonferroni_upper':upper[i]})
                        for alpha in ALPHAS:
                            bonfrej = (n > 0) & (upper <= alpha)
                            for method in methods:
                                index, pvals, levels, rejected = select(n, k, alpha, method, thresholds)
                                if method in ('bonferroni', 'tied_bonferroni'):
                                    assert np.array_equal(rejected, bonfrej)
                                if method in ('holm', 'fallback', 'tied_fallback'):
                                    assert np.all(rejected[bonfrej])
                                threshold = None if index is None else float(thresholds[index])
                                gamma = None if index is None else float(gammas[index])
                                point_upper = (np.nan if index is None else
                                               float(upper_bound(k[index:index+1],n[index:index+1],levels[index])[0]))
                                if index is not None: assert point_upper <= alpha+1e-12
                                row = info | {'alpha':alpha, 'method':method, 'supported':index is not None,
                                    'threshold':threshold, 'gamma':gamma,
                                    'selection_n_answered':int(n[index]) if index is not None else 0,
                                    'selection_n_incorrect':int(k[index]) if index is not None else 0,
                                    'selection_pvalue':float(pvals[index]) if index is not None else np.nan,
                                    'selection_test_level':float(levels[index]) if index is not None else np.nan,
                                    'selected_point_upper_at_test_level':point_upper,
                                    'guarantee_scope':'all_alpha' if method.endswith('bonferroni') else 'fixed_alpha',
                                    **test_metrics(nt, kt, len(test), index),
                                    **expected_test(u[test],z[test],threshold,gamma)}
                                row |= {'common_'+key:value for key,value in test_metrics(nc,kc,len(common),index).items()}
                                if method == 'bonferroni':
                                    reference = references.loc[(grader, model, allocation, seed, alpha)]
                                    for field in ['threshold','n_answered','n_incorrect','answer_rate','observed_error',
                                                  'common_n_answered','common_n_incorrect','common_observed_error']:
                                        a, b = row[field], reference[field]
                                        assert (pd.isna(a) and pd.isna(b)) or abs(a-b)<1e-11, (info,alpha,field,a,b)
                                    checked_results += 1
                                rows.append(row)
        if partition_index % 10 == 0:
            print(f'{args.stage}: finished partition {partition_index+1}/101', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/f'{args.stage}_all_results.csv',index=False)
    frame[frame.seed == prior.SEED].to_csv(OUT/f'{args.stage}_primary_results.csv',index=False)
    pd.DataFrame(primary_bounds).to_csv(OUT/f'{args.stage}_primary_bounds.csv',index=False)
    summaries(frame).to_csv(OUT/f'{args.stage}_summary.csv',index=False)
    baseline = (pd.read_csv(OUT/'deterministic_all_results.csv') if tied else frame)
    baseline = baseline[baseline.method == 'bonferroni']
    paired_summary(frame,baseline).to_csv(OUT/f'{args.stage}_paired.csv',index=False)
    metadata = {'stage':args.stage, 'seed':prior.SEED, 'additional_partitions':100,
                'alphas':ALPHAS,'delta':DELTA,'threshold_grid':grid.tolist(),
                'gammas':GAMMAS if tied else [1.], 'coin_replicates':5 if tied else 0,
                'methods':methods, 'source_inputs':sources,
                'baseline_bounds_reproduced':checked_bounds, 'baseline_results_reproduced':checked_results,
                'source_sha256':prior.sha(__file__),
                'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                'interpretation':'Exploratory, overlapping partitions; no new external validation. Test error above alpha is not a guarantee-failure estimate.'}
    (OUT/f'{args.stage}_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Wrote {len(rows):,} rows to {OUT}',flush=True)


if __name__ == '__main__':
    main()
