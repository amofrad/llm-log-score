"""Compare fixed-grid threshold decisions under 50–50 and 70–30 allocations.

Reproduce the sample-allocation comparison and export Table S10.
Uses saved reports and correctness grades; no model or grading calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260912
REPEATS = 100
ALPHAS = (.20, .25, .30, .35, .40)
DELTA = .05
MODELS = {'gemini35flash': 'Gemini', 'sonnet46': 'Sonnet', 'deepseekv32maas': 'DeepSeek'}
GRADERS = {'openai': 'GPT-5.6 Terra (primary)', 'gemini': 'Gemini 3.5 Flash (second)'}
ALLOCATIONS = {'50-50': 5, '70-30': 7}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_indices(n, seed, tenths):
    if tenths not in (5, 7):
        raise ValueError('Only the prespecified 50–50 and 70–30 allocations are supported')
    order = np.random.default_rng(seed).permutation(n)
    cut = n * tenths // 10
    return order[:cut], order[cut:], order[n * 7 // 10:]


def bounds_at(u, z, grid):
    selected = u[:, None] >= grid[None, :]
    n = selected.sum(axis=0)
    k = (selected * (1 - z[:, None])).sum(axis=0)
    upper = np.ones(len(grid))
    valid = k < n
    upper[valid] = beta.ppf(1 - DELTA / len(grid), k[valid] + 1, n[valid] - k[valid])
    return n, k, upper


def choose(grid, n, upper, alpha):
    qualifying = np.flatnonzero((n > 0) & (upper <= alpha))
    return int(qualifying[0]) if len(qualifying) else None


def evaluate(u, z, threshold):
    selected = np.zeros(len(u), dtype=bool) if threshold is None else u >= threshold
    n = int(selected.sum())
    k = int((1-z[selected]).sum())
    return {'n_test': len(u), 'n_answered': n, 'n_incorrect': k,
            'answer_rate': n / len(u), 'observed_error': k / n if n else None,
            'error_lo': (0. if k == 0 else float(beta.ppf(.025, k, n-k+1))) if n else None,
            'error_hi': (1. if k == n else float(beta.ppf(.975, k+1, n-k))) if n else None}


def equal(value, reference):
    if value is None:
        return reference == ''
    return abs(float(value) - float(reference)) < 1e-11


def load_grader(grader, analysis_dir=None):
    analysis_dir = Path(analysis_dir) if analysis_dir else ROOT / 'results/intermediate'
    folder = analysis_dir / f'threshold_selection_{grader}'
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['seed'] == SEED and meta['additional_splits'] == REPEATS and meta['delta'] == DELTA
    for entry in meta['inputs']:
        assert sha(ROOT/entry['file']) == entry['sha256'], 'Source-response hash changed'
    raw = [r for r in read_csv(folder/'primary_inputs.csv') if r['representation'] == 'raw']
    frames = {}
    for model in MODELS:
        rows = sorted((r for r in raw if r['model_key'] == model), key=lambda r:r['question_id'])
        ids = [r['question_id'] for r in rows]
        assert len(ids) == len(set(ids)) == 4326
        u = np.array([float(r['u']) for r in rows])
        z = np.array([int(r['z']) for r in rows])
        assert np.all((u == -1) | ((u >= 0) & (u <= 1))) and np.all(np.isin(z, [0, 1]))
        assert np.all(z[u == -1] == 0)
        frames[model] = (ids, u, z)
    references = {}
    for r in read_csv(folder/'all_results.csv'):
        if r['representation'] == 'raw' and r['method'] == 'fixed_full':
            alpha = float(r['alpha'])
            if any(abs(alpha-a)<1e-12 for a in ALPHAS):
                key = (r['model_key'], int(r['seed']), round(alpha, 12))
                assert key not in references
                references[key] = r
    bound_refs = {}
    for r in read_csv(folder/'all_bounds.csv'):
        if r['representation'] == 'raw' and r['method'] == 'fixed_full':
            key = (r['model_key'], int(r['seed']), float(r['threshold']))
            assert key not in bound_refs
            bound_refs[key] = r
    assert len(references) == 3 * 101 * 5 and len(bound_refs) == 3 * 101 * 102
    return frames, np.array(meta['fixed_grid']), references, bound_refs, {
        'grader': grader, 'source_metadata_sha256': sha(folder/'metadata.json'),
        'per_question_inputs_sha256': sha(folder/'primary_inputs.csv'),
        'original_results_sha256': sha(folder/'all_results.csv'),
        'original_bounds_sha256': sha(folder/'all_bounds.csv'), 'responses': meta['inputs']}


def summarize(rows):
    summaries, paired = [], []
    for grader in GRADERS:
        for model in MODELS:
            for alpha in ALPHAS:
                groups = {}
                for allocation in ALLOCATIONS:
                    group = [r for r in rows if r['grader']==grader and r['model_key']==model
                             and r['alpha']==alpha and r['allocation']==allocation and r['seed']!=SEED]
                    assert len(group)==REPEATS
                    groups[allocation] = {r['seed']:r for r in group}
                    supported = [r for r in group if r['supported']]
                    answering = [r for r in group if r['n_answered']]
                    errors = [r['observed_error'] for r in answering]
                    widths = [r['error_hi']-r['error_lo'] for r in answering]
                    summaries.append({'grader':grader,'model_key':model,'alpha':alpha,'allocation':allocation,
                        'n_splits':len(group), 'n_supported':len(supported),
                        'mean_answer_rate':float(np.mean([r['answer_rate'] for r in group])),
                        'mean_common_answer_rate':float(np.mean([r['common_answer_rate'] for r in group])),
                        'median_threshold_when_supported':float(np.median([r['threshold'] for r in supported])) if supported else None,
                        'n_splits_with_test_answers':len(answering),
                        'median_test_error_when_answering':float(np.median(errors)) if errors else None,
                        'test_error_q25':float(np.quantile(errors,.25)) if errors else None,
                        'test_error_q75':float(np.quantile(errors,.75)) if errors else None,
                        'mean_test_interval_width':float(np.mean(widths)) if widths else None,
                        'n_test_errors_above_alpha':sum(e>alpha for e in errors),
                        'median_minimum_supported_alpha':float(np.median([r['minimum_supported_alpha'] for r in group]))})
                pairs = [(groups['50-50'][seed],groups['70-30'][seed]) for seed in sorted(groups['50-50'])]
                both = [(a,b) for a,b in pairs if a['supported'] and b['supported']]
                common_answering = [(a,b) for a,b in pairs if a['common_n_answered'] and b['common_n_answered']]
                deltas = [b['common_answer_rate']-a['common_answer_rate'] for a,b in pairs]
                paired.append({'grader':grader,'model_key':model,'alpha':alpha,
                    'n_support_gained':sum(not a['supported'] and b['supported'] for a,b in pairs),
                    'n_support_lost':sum(a['supported'] and not b['supported'] for a,b in pairs),
                    'n_both_supported':len(both),
                    'n_70_lower_threshold_when_both_supported':sum(b['threshold']<a['threshold'] for a,b in both),
                    'n_70_higher_threshold_when_both_supported':sum(b['threshold']>a['threshold'] for a,b in both),
                    'mean_common_answer_rate_change':float(np.mean(deltas)),
                    'common_answer_rate_change_q25':float(np.quantile(deltas,.25)),
                    'common_answer_rate_change_q75':float(np.quantile(deltas,.75)),
                    'n_both_answer_on_common_test':len(common_answering),
                    'mean_common_error_change_when_both_answer':float(np.mean([b['common_observed_error']-a['common_observed_error'] for a,b in common_answering])) if common_answering else None})
    return summaries, paired


def run(out, analysis_dir=None):
    frames_by_grader, inputs, refs, bound_refs = {}, [], {}, {}
    grid = ids = None
    for grader in GRADERS:
        frames,g,ref,bref,metadata = load_grader(grader, analysis_dir)
        if grid is not None: np.testing.assert_array_equal(g,grid)
        grid=g
        for model,(model_ids,u,z) in frames.items():
            if ids is not None: assert model_ids==ids
            ids=model_ids
            if grader=='gemini':np.testing.assert_array_equal(u,frames_by_grader['openai'][model][1])
        frames_by_grader[grader]=frames;refs[grader]=ref;bound_refs[grader]=bref;inputs.append(metadata)
    assert len(grid)==102 and np.all(np.diff(grid)>0)
    results, bands, assignments = [], [], []
    for seed in range(SEED,SEED+REPEATS+1):
        for allocation,tenths in ALLOCATIONS.items():
            sel,test,common=split_indices(len(ids),seed,tenths)
            assert len(set(sel)&set(test))==0 and set(common)<=set(test)
            if seed==SEED:
                for i,q in enumerate(ids):
                    assignments.append({'question_id':q,'allocation':allocation,
                                        'split':'selection' if i in set(sel) else 'test',
                                        'common_test':i in set(common)})
            for grader in GRADERS:
                for model in MODELS:
                    _,u,z=frames_by_grader[grader][model]
                    n,k,upper=bounds_at(u[sel],z[sel],grid)
                    for j,t in enumerate(grid):
                        bands.append({'grader':grader,'model_key':model,'allocation':allocation,'seed':seed,
                                      'threshold':float(t),'n_selection':len(sel),'n_answered':int(n[j]),
                                      'n_incorrect':int(k[j]),'risk_upper':float(upper[j])})
                        if allocation=='50-50':
                            ref=bound_refs[grader][(model,seed,float(t))]
                            assert int(n[j])==int(ref['n_cert_answered']) and int(k[j])==int(ref['k_cert_incorrect'])
                            assert equal(upper[j],ref['risk_upper'])
                    minimum=float(upper[n>0].min()) if np.any(n>0) else 1.
                    previous=float('inf')
                    for alpha in ALPHAS:
                        j=choose(grid,n,upper,alpha)
                        t=float(grid[j]) if j is not None else None
                        assert (float('inf') if t is None else t)<=previous
                        previous=float('inf') if t is None else t
                        row={'grader':grader,'model_key':model,'allocation':allocation,'seed':seed,'alpha':alpha,
                             'n_selection':len(sel),'supported':j is not None,'threshold':t,
                             'selection_n_answered':int(n[j]) if j is not None else 0,
                             'selection_n_incorrect':int(k[j]) if j is not None else 0,
                             'selection_upper':float(upper[j]) if j is not None else None,
                             'minimum_supported_alpha':minimum,**evaluate(u[test],z[test],t),
                             **{'common_'+k:v for k,v in evaluate(u[common],z[common],t).items()}}
                        if allocation=='50-50':
                            ref=refs[grader][(model,seed,round(alpha,12))]
                            assert row['supported']==(ref['supported']=='True')
                            for field in ['threshold','n_test','n_answered','n_incorrect','answer_rate','observed_error','error_lo','error_hi']:
                                assert equal(row[field],ref[field]),(grader,model,seed,alpha,field)
                            for field,old in [('selection_upper','certification_upper'),('selection_n_answered','n_cert_answered'),('selection_n_incorrect','k_cert_incorrect')]:
                                assert equal(row[field],ref[old])
                        results.append(row)
        if (seed-SEED)%20==0:print(f'Completed split {seed-SEED+1}/101',flush=True)
    summary,paired=summarize(results)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('all_results',results),('all_bounds',bands),('primary_results',[r for r in results if r['seed']==SEED]),('split_summary',summary),('paired_comparisons',paired),('primary_assignments',assignments)]:
        write_csv(out/(name+'.csv'),rows)
    meta={'seed':SEED,'additional_splits':REPEATS,'alphas':ALPHAS,'delta':DELTA,'grid':grid.tolist(),
          'selection_sizes':{'50-50':2163,'70-30':3028},'test_sizes':{'50-50':2163,'70-30':1298},
          'primary_grader':'openai','representation':'raw','common_test_size':1298,
          'verification':{'original_bounds_reproduced':61812,'original_result_rows_reproduced':3030},
          'inputs':inputs,'source_sha256':sha(__file__),
          'software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Wrote {len(results)} result rows and {len(bands)} bounds to {out}',flush=True)



def summarize_errors(rows):
    errors=[]
    for row in rows:
        n=int(row['common_n_answered'])
        if n:
            error=float(row['common_observed_error'])
            assert math.isclose(error,int(row['common_n_incorrect'])/n)
            errors.append(error)
        else:
            assert row['common_observed_error']==''
    if not errors:
        return {'n_partitions_with_test_answers':0,'median_common_test_error':None,
                'common_test_error_q25':None,'common_test_error_q75':None}
    median=statistics.median(errors)
    q25,_,q75=statistics.quantiles(errors,n=4,method='inclusive') if len(errors)>1 else [median]*3
    return {'n_partitions_with_test_answers':len(errors),'median_common_test_error':median,
            'common_test_error_q25':q25,'common_test_error_q75':q75}


def export_table(source, out, primary_grader='openai'):
    if primary_grader not in ('openai','gemini'):
        raise ValueError('Unknown grader: '+primary_grader)
    meta=json.loads((source/'metadata.json').read_text())
    assert meta['additional_splits']==100 and meta['common_test_size']==1298
    assert meta['verification']['original_result_rows_reproduced']==3030
    with (source/'split_summary.csv').open() as stream: rows=list(csv.DictReader(stream))
    lookup={(r['model_key'],float(r['alpha']),r['grader'],r['allocation']):r for r in rows}
    assert len(lookup)==60
    groups={key:[] for key in lookup}
    with (source/'all_results.csv').open() as stream:
        for row in csv.DictReader(stream):
            if int(row['seed'])==meta['seed']:continue
            key=(row['model_key'],float(row['alpha']),row['grader'],row['allocation'])
            groups[key].append(row)
    assert all(len(group)==len({r['seed'] for r in group})==100 for group in groups.values())
    records=[]
    for model in MODELS:
        for alpha in ALPHAS:
            record={'model_key':model,'alpha':alpha,'display_grader':primary_grader}
            for grader in ['openai','gemini']:
                pair=[lookup[model,alpha,grader,a] for a in ['50-50','70-30']]
                assert all(int(r['n_splits'])==100 for r in pair)
                for allocation,r in zip(['50-50','70-30'],pair):
                    group=groups[model,alpha,grader,allocation]
                    assert all(int(x['common_n_test'])==1298 for x in group)
                    summary=summarize_errors(group)
                    assert summary['n_partitions_with_test_answers']==int(r['n_supported'])
                    assert math.isclose(statistics.mean(float(x['common_answer_rate']) for x in group),
                                        float(r['mean_common_answer_rate']),abs_tol=1e-12)
                    record[f'{grader}_{allocation}_n_supported']=int(r['n_supported'])
                    record[f'{grader}_{allocation}_mean_common_answer_rate']=float(r['mean_common_answer_rate'])
                    record.update({f'{grader}_{allocation}_{key}':value for key,value in summary.items()})
            records.append(record)
    out.mkdir(parents=True,exist_ok=True)
    stem='TableS10_SplitSensitivity'
    with (out/(stem+'.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('Generated '+str(out/(stem+'.csv')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/figures")
    parser.add_argument("--grader", choices=("openai", "gemini"), default="openai",
                        help="Grader displayed in Table S10; the CSV retains both graders.")
    parser.add_argument("--analysis-dir", type=Path,
                        help="Keep intermediate results here; otherwise use a temporary directory.")
    args = parser.parse_args()

    def reproduce(analysis_dir):
        from threshold_selection import analysis_complete, run_analysis
        for grader in GRADERS:
            folder = analysis_dir / f"threshold_selection_{grader}"
            if not analysis_complete(folder):
                run_analysis(grader, folder, repeats=REPEATS)
        source = analysis_dir / "threshold_split_sensitivity"
        run(source, analysis_dir=analysis_dir)
        export_table(source, args.out_dir, primary_grader=args.grader)

    if args.analysis_dir is None:
        with tempfile.TemporaryDirectory(prefix="rbd-split-") as temporary:
            reproduce(Path(temporary))
    else:
        reproduce(args.analysis_dir)


if __name__ == "__main__":
    main()
